pub mod rpc;

use std::{future::Future, path::PathBuf, sync::Arc};

use atomic_counter::AtomicCounter;

/// Defines how to spawn a model.
#[derive(Clone, Debug, PartialEq)]
pub enum ModelToExecute<'a, S: AsRef<str>> {
    Bin { path: &'a str },
    NodeJs { flags: &'a [S] },
    Python { flags: &'a [S] },
}

/// Executes child processes and sets up communications to those using Unix sockets.
#[derive(Clone)]
pub struct ModelExecutor {
    temp_dir: PathBuf,
    model_counter: Arc<atomic_counter::ConsistentCounter>,
}

impl ModelExecutor {
    pub fn new(temp_dir: PathBuf) -> Self {
        Self {
            temp_dir,
            model_counter: Arc::new(atomic_counter::ConsistentCounter::new(0)),
        }
    }

    /// Lower lever API for spawning Decthings models. You most likely want to use *run_bin*,
    /// *run_node_js* or *run_python* instead.
    ///
    /// Returns a tokio::process::Command which has been setup to spawn the model. Also sets up a
    /// Unix socket listener, and returns a Future which resolves once the child has connected to
    /// us using the Unix socket. Using this connection you can call RPC methods and listen for
    /// events on the child.
    ///
    /// Before you spawn the command you can add additional environment variables, arguments and
    /// other options if necessary.
    ///
    /// For Node.js and Python, you should call the `Initialize` command on the child to specify
    /// the .js or .py file to execute.
    pub fn run<S: AsRef<str>>(
        &self,
        model: ModelToExecute<'_, S>,
    ) -> (
        tokio::process::Command,
        impl Future<Output = (rpc::ChildRpc, rpc::ChildRpcListener)>,
    ) {
        let model_count = self.model_counter.inc();
        let unixsocket_path = self.temp_dir.join(model_count.to_string());

        let mut command = match model {
            ModelToExecute::Bin { path } => {
                let mut command = tokio::process::Command::new(path);
                command.env_clear();
                command
            }
            ModelToExecute::NodeJs { flags } => {
                let mut command = tokio::process::Command::new("bash");
                command.args([
                    "-c",
                    &format!(
                        r#"P=$(which decthings-model-node)
if [ $? -eq 0 ]; then
    exec node {} $P
else
    echo "Could not find command decthings-model-node. Install it using 'npm i -g @decthings/model'." 1>&2
    exit 1
fi"#,
                        flags
                            .iter()
                            .map(|x| x.as_ref())
                            .collect::<Vec<_>>()
                            .join(" ")
                    ),
                ]);
                command.env_clear();
                command
            }
            ModelToExecute::Python { flags } => {
                let mut command = tokio::process::Command::new("python3");
                command.env_clear();
                for flag in flags {
                    command.arg(flag.as_ref());
                }
                command.args(["-m", "decthings_model"]);
                command
            }
        };
        command.env("IPC_PATH", &unixsocket_path);

        let connection_future = rpc::rpc(unixsocket_path, model_count);

        (command, connection_future)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::{os::unix::process::ExitStatusExt, process::Stdio};
    use tokio::io::AsyncReadExt;

    #[tokio::test]
    async fn test_js_child_hello_world() {
        let _ = env_logger::builder().is_test(true).try_init();

        let temp_dir = tempdir::TempDir::new("decthings-execute-model-js-hello-world").unwrap();
        let temp_dir = temp_dir.path().to_owned();

        tokio::fs::write(
            temp_dir.join("test.js"),
            "console.log('Hello world from stdout!')
console.error('Hello world from stderr!')
setTimeout(() => { process.exit(12) }, 1000)
exports.default = {}",
        )
        .await
        .unwrap();

        let model_executor = ModelExecutor::new(temp_dir.clone());
        let (mut command, rpc_fut) =
            model_executor.run::<&str>(ModelToExecute::NodeJs { flags: &[] });

        #[allow(dead_code)]
        fn require_send<T: Send>(_t: &T) {}

        require_send(&rpc_fut);

        command.env("PATH", std::env::var("PATH").unwrap());
        command.stdout(Stdio::piped()).stderr(Stdio::piped());

        let mut child = command.spawn().unwrap();

        let mut stdout = child.stdout.take().unwrap();
        let mut stderr = child.stderr.take().unwrap();

        tokio::join!(
            async {
                let (rpc, rpc_listener) = rpc_fut.await;

                tokio::join!(
                    async {
                        let fut = rpc.call_initialize(&rpc::types::InitializeCommand {
                            path: temp_dir.join("test.js").display().to_string(),
                        });
                        require_send(&fut);
                        fut.await.unwrap();
                    },
                    async {
                        struct Callback {
                            did_initialize: Arc<std::sync::Mutex<bool>>,
                        }
                        impl rpc::ChildEventCallbacks for Callback {
                            fn on_event<'a>(
                                &'a self,
                                event: rpc::types::EventMessage,
                                blobs: Box<dyn crate::unix::Blobs + Send + 'a>,
                            ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>
                            {
                                Box::pin(async move {
                                    assert_eq!(
                                        event,
                                        rpc::types::EventMessage::ModelSessionInitialized {
                                            error: None,
                                        }
                                    );
                                    assert_eq!(blobs.amount(), 0);
                                    let mut locked = self.did_initialize.lock().unwrap();
                                    assert!(!*locked);
                                    *locked = true;
                                })
                            }

                            fn on_data_event<'a>(
                                &'a self,
                                _event: rpc::types::DataEvent,
                            ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>
                            {
                                unreachable!()
                            }
                        }
                        let did_initialize = Arc::new(std::sync::Mutex::new(false));
                        let listen_fut = rpc_listener.listen(Callback {
                            did_initialize: Arc::clone(&did_initialize),
                        });
                        require_send(&listen_fut);
                        listen_fut.await;
                        assert!(*did_initialize.lock().unwrap());
                    }
                );
            },
            async {
                let t0 = std::time::Instant::now();
                let wait_result = child.wait().await.unwrap();
                let elapsed = t0.elapsed().as_millis();
                assert!(elapsed > 1000 && elapsed < 1500);
                assert_eq!(wait_result.code(), Some(12));
                assert_eq!(wait_result.signal(), None);
            },
            async {
                let mut stdout_data: Vec<u8> = vec![];
                loop {
                    let mut buf = vec![0; 1024];
                    let amount_read: usize = stdout.read(&mut buf).await.unwrap();
                    if amount_read == 0 {
                        break;
                    }
                    stdout_data.extend_from_slice(&buf[0..amount_read]);
                }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                println!("{}", String::from_utf8_lossy(&stdout_data));
                assert_eq!(stdout_data, b"Hello world from stdout!\n");
            },
            async {
                let mut stderr_data: Vec<u8> = vec![];
                loop {
                    let mut buf = vec![0; 1024];
                    let amount_read: usize = stderr.read(&mut buf).await.unwrap();
                    if amount_read == 0 {
                        break;
                    }
                    stderr_data.extend_from_slice(&buf[0..amount_read]);
                }
                println!("{}", String::from_utf8_lossy(&stderr_data));
                assert_eq!(stderr_data, b"Hello world from stderr!\n");
            }
        );
    }
}
