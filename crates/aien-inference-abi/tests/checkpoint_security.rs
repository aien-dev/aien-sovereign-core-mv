//! Offline real-loader regressions, also runnable against origin/main.
use aien_inference_abi::checkpoint::{checkpoint_files, load_checkpoint_with_catalog};
use std::{fs, path::Path};

fn shard(path: &Path) {
    let header = br#"{"w":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&[0, 0]);
    fs::write(path, bytes).unwrap();
}

#[test]
fn index_cannot_load_absolute_parent_or_symlink_shards() {
    let temp = std::env::temp_dir().join(format!("sc-sec-5-{}", std::process::id()));
    fs::create_dir(&temp).unwrap();
    let model = temp.join("model");
    fs::create_dir(&model).unwrap();
    shard(&temp.join("outside.safetensors"));
    shard(&model.join("inside.safetensors"));
    std::os::unix::fs::symlink(temp.join("outside.safetensors"), model.join("link.safetensors")).unwrap();
    let catalog = vec![("w".into(), vec![1])];
    let mut accepted = Vec::new();
    for name in [temp.join("outside.safetensors").display().to_string(), "../outside.safetensors".into(), "link.safetensors".into()] {
        fs::write(model.join("model.safetensors.index.json"), serde_json::json!({"weight_map":{"w":name}}).to_string()).unwrap();
        if load_checkpoint_with_catalog(&model, &catalog).is_ok() { accepted.push(name); }
    }
    // Negative control: a regular shard in the checkpoint directory must load.
    fs::write(model.join("model.safetensors.index.json"), r#"{"weight_map":{"w":"inside.safetensors"}}"#).unwrap();
    assert!(load_checkpoint_with_catalog(&model, &catalog).is_ok());
    fs::remove_dir_all(temp).unwrap();
    assert!(accepted.is_empty(), "escaped index shards were loaded: {accepted:?}");
}

#[test]
fn oversized_and_nonregular_files_are_rejected_before_reads() {
    let temp = std::env::temp_dir().join(format!("sc-sec-5-bounds-{}", std::process::id()));
    fs::create_dir(&temp).unwrap();
    let sparse = temp.join("model.safetensors");
    let file = fs::File::create(&sparse).unwrap();
    // Filesystem-safe sparse size exceeds the documented production shard cap.
    file.set_len(274877906945).unwrap();
    let oversized_accepted = checkpoint_files(&sparse).is_ok();
    fs::remove_file(&sparse).unwrap();
    std::os::unix::fs::symlink("/dev/null", &sparse).unwrap();
    let nonregular_accepted = checkpoint_files(&sparse).is_ok();
    fs::remove_dir_all(temp).unwrap();
    assert!(!oversized_accepted, "unbounded file length accepted");
    assert!(!nonregular_accepted, "nonregular/symlink input accepted");
}
