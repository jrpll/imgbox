use std::sync::Mutex;
use tauri::{AppHandle, Manager};
use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::CommandChild;

pub struct SidecarState(pub Mutex<Option<CommandChild>>);

pub fn start(app: &AppHandle, hf_token: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let binding = app.state::<SidecarState>();
    let mut state = binding.0.lock().unwrap();

    if state.is_some() {
        return Ok(());
    }

    let mut cmd = app.shell().sidecar("imgbox-server")?;

    if let Some(token) = hf_token {
        cmd = cmd.env("HUGGING_FACE_TOKEN", token);
    }

    let (_rx, child) = cmd.spawn()?;
    *state = Some(child);

    log::info!("imgbox-server sidecar started");
    Ok(())
}

pub fn stop(app: &AppHandle) {
    let binding = app.state::<SidecarState>();
    let mut state = binding.0.lock().unwrap();
    if let Some(child) = state.take() {
        let _ = child.kill();
        log::info!("imgbox-server sidecar stopped");
    }
}
