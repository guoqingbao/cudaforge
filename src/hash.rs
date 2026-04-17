//! Build cache for incremental compilation

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
// redundant walkdir import removed

const CACHE_FILENAME: &str = ".cudaforge_cache.json";

/// Build cache for tracking file modifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildCache {
    /// Cached entries keyed by source file path
    entries: HashMap<String, CacheEntry>,
    /// Version of the cache format
    version: u32,
}

/// Entry for a single source file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// SHA-256 hash of file content
    pub content_hash: String,
    /// Combined hash of watched paths
    #[serde(default)]
    pub watch_hash: String,
    /// Last modification time (Unix timestamp)
    pub modified_time: u64,
    /// Path to compiled object file
    pub object_path: String,
    /// GPU architecture used for compilation (e.g., "sm_90a", "sm_80")
    pub gpu_arch: String,
    /// Extra args used (hash of args)
    pub args_hash: String,
}

impl Default for BuildCache {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            version: 1,
        }
    }
}

impl BuildCache {
    /// Load cache from build directory
    pub fn load(build_dir: &Path) -> Self {
        let cache_path = build_dir.join(CACHE_FILENAME);

        if cache_path.exists() {
            if let Ok(contents) = fs::read_to_string(&cache_path) {
                if let Ok(cache) = serde_json::from_str::<BuildCache>(&contents) {
                    return cache;
                }
            }
        }

        Self::default()
    }

    /// Save cache to build directory
    pub fn save(&self, build_dir: &Path) -> Result<()> {
        let cache_path = build_dir.join(CACHE_FILENAME);
        let contents = serde_json::to_string_pretty(self)
            .map_err(|e| Error::CacheError(format!("Failed to serialize cache: {}", e)))?;

        fs::write(&cache_path, contents)
            .map_err(|e| Error::CacheError(format!("Failed to write cache: {}", e)))?;

        Ok(())
    }

    /// Check if a file needs recompilation
    pub fn needs_rebuild(
        &self,
        source_path: &Path,
        object_path: &Path,
        gpu_arch: &str,
        args_hash: &str,
        watch_hash: &str,
    ) -> bool {
        let key = format!("{}:{}", source_path.display(), object_path.display());

        // Check if object file exists
        if !object_path.exists() {
            return true;
        }

        // Get cached entry
        let entry = match self.entries.get(&key) {
            Some(e) => e,
            None => return true,
        };

        // Check if gpu arch, args, or watched paths changed
        if entry.gpu_arch != gpu_arch
            || entry.args_hash != args_hash
            || entry.watch_hash != watch_hash
        {
            return true;
        }

        // Check content hash
        if let Ok(current_hash) = hash_file(source_path) {
            if current_hash != entry.content_hash {
                return true;
            }
        } else {
            return true;
        }

        // Check object path matches
        if entry.object_path != object_path.to_string_lossy() {
            return true;
        }

        false
    }

    /// Update cache entry for a compiled file
    pub fn update(
        &mut self,
        source_path: &Path,
        object_path: &Path,
        gpu_arch: &str,
        args_hash: &str,
        watch_hash: &str,
    ) -> Result<()> {
        let key = format!("{}:{}", source_path.display(), object_path.display());
        let content_hash = hash_file(source_path)?;

        let modified_time = source_path
            .metadata()
            .and_then(|m| m.modified())
            .map(|t| {
                t.duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
            })
            .unwrap_or(0);

        self.entries.insert(
            key,
            CacheEntry {
                content_hash,
                watch_hash: watch_hash.to_string(),
                modified_time,
                object_path: object_path.to_string_lossy().to_string(),
                gpu_arch: gpu_arch.to_string(),
                args_hash: args_hash.to_string(),
            },
        );

        Ok(())
    }

    /// Remove stale entries (files that no longer exist)
    pub fn cleanup(&mut self) {
        self.entries.retain(|key, entry| {
            let source_exists = source_path_from_key(key, &entry.object_path)
                .map(Path::new)
                .is_some_and(Path::exists);

            source_exists && Path::new(&entry.object_path).exists()
        });
    }
}

fn source_path_from_key<'a>(key: &'a str, object_path: &str) -> Option<&'a str> {
    let suffix = format!(":{}", object_path);
    key.strip_suffix(&suffix)
}

/// Compute SHA-256 hash of a file's contents
pub fn hash_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

/// Hash a list of arguments for cache comparison
pub fn hash_args(args: &[String]) -> String {
    let mut hasher = Sha256::new();
    for arg in args {
        hasher.update(arg.as_bytes());
        hasher.update(b"\0");
    }
    format!("{:x}", hasher.finalize())
}

/// Compute a combined hash of multiple paths (files or directories)
pub fn hash_paths(paths: &[PathBuf]) -> String {
    let mut hasher = Sha256::new();

    // Sort paths to ensure deterministic hashing
    let mut sorted_paths = paths.to_vec();
    sorted_paths.sort();

    for path in sorted_paths {
        if path.is_file() {
            if let Ok(h) = hash_file(&path) {
                hasher.update(path.to_string_lossy().as_bytes());
                hasher.update(b":");
                hasher.update(h.as_bytes());
                hasher.update(b"\0");
            }
        } else if path.is_dir() {
            // For directories, we hash all .h and .cuh files recursively
            let mut entries: Vec<_> = walkdir::WalkDir::new(&path)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| {
                    let p = e.path();
                    p.is_file()
                        && matches!(
                            p.extension().and_then(|s| s.to_str()),
                            Some("h" | "cuh" | "hpp")
                        )
                })
                .collect();

            entries.sort_by(|a, b| a.path().cmp(b.path()));

            for entry in entries {
                if let Ok(h) = hash_file(entry.path()) {
                    hasher.update(entry.path().to_string_lossy().as_bytes());
                    hasher.update(b":");
                    hasher.update(h.as_bytes());
                    hasher.update(b"\0");
                }
            }
        }
    }

    format!("{:x}", hasher.finalize())
}

/// Check if output file is newer than all input files
#[allow(dead_code)]
pub fn output_is_current(output: &Path, inputs: &[PathBuf]) -> bool {
    let output_modified = match output.metadata().and_then(|m| m.modified()) {
        Ok(t) => t,
        Err(_) => return false,
    };

    for input in inputs {
        let input_modified = match input.metadata().and_then(|m| m.modified()) {
            Ok(t) => t,
            Err(_) => return false,
        };

        if input_modified.duration_since(output_modified).is_ok() {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_hash_args() {
        let args1 = vec!["-O3".to_string(), "-std=c++17".to_string()];
        let args2 = vec!["-O3".to_string(), "-std=c++17".to_string()];
        let args3 = vec!["-O2".to_string(), "-std=c++17".to_string()];

        assert_eq!(hash_args(&args1), hash_args(&args2));
        assert_ne!(hash_args(&args1), hash_args(&args3));
    }

    #[test]
    fn test_cleanup_retains_valid_composite_key_entries() {
        let mut root = std::env::temp_dir();
        root.push(format!("cudaforge-hash-test-{}", std::process::id()));

        if root.exists() {
            let _ = fs::remove_dir_all(&root);
        }
        fs::create_dir_all(&root).unwrap();

        let source_path = root.join("kernel.cu");
        let object_path = root.join("kernel.o");
        fs::write(&source_path, "__global__ void kernel() {}").unwrap();
        fs::write(&object_path, "object").unwrap();

        let mut cache = BuildCache::default();
        cache
            .update(&source_path, &object_path, "sm_80", "args", "watch")
            .unwrap();

        cache.cleanup();
        assert_eq!(cache.entries.len(), 1);

        fs::remove_file(&source_path).unwrap();
        cache.cleanup();
        assert!(cache.entries.is_empty());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn test_source_path_from_key_with_colons_in_paths() {
        let key = "/tmp/src:dir/kernel.cu:/tmp/out:dir/kernel.o";
        let object_path = "/tmp/out:dir/kernel.o";

        assert_eq!(
            source_path_from_key(key, object_path),
            Some("/tmp/src:dir/kernel.cu")
        );
    }
}
