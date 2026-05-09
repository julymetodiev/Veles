fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use the vendored `protoc` binary so a system-wide install of the
    // protobuf compiler is not a prerequisite of `cargo install veles-grpc`
    // (or `veles-cli`, which depends on us).
    if std::env::var_os("PROTOC").is_none() {
        let protoc = protoc_bin_vendored::protoc_bin_path()?;
        // SAFETY: build scripts run single-threaded before any user code.
        unsafe {
            std::env::set_var("PROTOC", protoc);
        }
    }

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["proto/veles.proto"], &["proto/"])?;
    Ok(())
}
