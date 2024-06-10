use wasmtime::component::{Resource, ResourceTable};

pub trait HostDataLoader: Send + 'static {
    fn id(&self) -> u32;
    fn read(&mut self, start_index: u32, amount: u32) -> Vec<Vec<u8>>;
    fn shuffle(&mut self, others: &[u32]);
}

pub type HostDataLoaderDyn = Box<dyn HostDataLoader>;

pub trait HostStateProvider: Send + 'static {
    fn provide(&self, data: Vec<(String, Vec<u8>)>);
}

pub type HostStateProviderDyn = Box<dyn HostStateProvider>;

pub trait HostStateLoader: Send + 'static {
    fn read(&self) -> Vec<u8>;
}

pub type HostStateLoaderDyn = Box<dyn HostStateLoader>;

pub trait HostTrainTracker: Send + 'static {
    fn progress(&self, progress: f32);
    fn metrics(&self, metrics: Vec<(String, Vec<u8>)>);
    fn is_cancelled(&self) -> bool;
}

pub type HostTrainTrackerDyn = Box<dyn HostTrainTracker>;

wasmtime::component::bindgen!({
    with: {
        "decthings:model/model-callbacks/data-loader": HostDataLoaderDyn,
        "decthings:model/model-callbacks/state-provider": HostStateProviderDyn,
        "decthings:model/model-callbacks/state-loader": HostStateLoaderDyn,
        "decthings:model/model-callbacks/train-tracker": HostTrainTrackerDyn,
    },
    trappable_imports: true,
});

pub struct Host {
    pub(super) table: ResourceTable,
    pub(super) wasi: wasmtime_wasi::WasiCtx,
}

impl wasmtime_wasi::WasiView for Host {
    fn table(&mut self) -> &mut ResourceTable {
        &mut self.table
    }

    fn ctx(&mut self) -> &mut wasmtime_wasi::WasiCtx {
        &mut self.wasi
    }
}

impl decthings::model::model_callbacks::HostDataLoader for Host {
    fn read(
        &mut self,
        self_: Resource<HostDataLoaderDyn>,
        start_index: u32,
        amount: u32,
    ) -> wasmtime::Result<Vec<Vec<u8>>> {
        debug_assert!(!self_.owned());
        let data_loader = self.table.get_mut(&self_)?;
        Ok(data_loader.read(start_index, amount))
    }

    fn shuffle(
        &mut self,
        self_: Resource<HostDataLoaderDyn>,
        others: Vec<Resource<HostDataLoaderDyn>>,
    ) -> wasmtime::Result<()> {
        debug_assert!(!self_.owned());
        let others: Vec<u32> = others
            .into_iter()
            .map(|resource| Ok(self.table.get(&resource)?.id()))
            .collect::<wasmtime::Result<_>>()?;
        let data_loader = self.table.get_mut(&self_)?;
        data_loader.shuffle(&others);
        Ok(())
    }

    fn drop(&mut self, rep: Resource<HostDataLoaderDyn>) -> wasmtime::Result<()> {
        debug_assert!(rep.owned());
        let _data_loader = self.table.delete(rep)?;
        Ok(())
    }
}

impl decthings::model::model_callbacks::HostStateProvider for Host {
    fn provide(
        &mut self,
        self_: Resource<HostStateProviderDyn>,
        data: Vec<(String, Vec<u8>)>,
    ) -> wasmtime::Result<()> {
        debug_assert!(!self_.owned());
        let state_provider = self.table.get(&self_)?;
        state_provider.provide(data);
        Ok(())
    }

    fn drop(&mut self, rep: Resource<HostStateProviderDyn>) -> wasmtime::Result<()> {
        debug_assert!(rep.owned());
        let _state_provider = self.table.delete(rep)?;
        Ok(())
    }
}

impl decthings::model::model_callbacks::HostStateLoader for Host {
    fn read(&mut self, self_: Resource<HostStateLoaderDyn>) -> wasmtime::Result<Vec<u8>> {
        debug_assert!(!self_.owned());
        let state_provider = self.table.get(&self_)?;
        Ok(state_provider.read())
    }

    fn drop(&mut self, rep: Resource<HostStateLoaderDyn>) -> wasmtime::Result<()> {
        debug_assert!(rep.owned());
        let _state_loader = self.table.delete(rep)?;
        Ok(())
    }
}

impl decthings::model::model_callbacks::HostTrainTracker for Host {
    fn progress(
        &mut self,
        self_: Resource<HostTrainTrackerDyn>,
        progress: f32,
    ) -> wasmtime::Result<()> {
        debug_assert!(!self_.owned());
        let train_tracker = self.table.get(&self_)?;
        train_tracker.progress(progress);
        Ok(())
    }

    fn metrics(
        &mut self,
        self_: Resource<HostTrainTrackerDyn>,
        metrics: Vec<(String, Vec<u8>)>,
    ) -> wasmtime::Result<()> {
        debug_assert!(!self_.owned());
        let train_tracker = self.table.get(&self_)?;
        train_tracker.metrics(metrics);
        Ok(())
    }

    fn is_cancelled(
        &mut self,
        self_: wasmtime::component::Resource<__with_name3>,
    ) -> wasmtime::Result<bool> {
        debug_assert!(!self_.owned());
        let train_tracker = self.table.get(&self_)?;
        Ok(train_tracker.is_cancelled())
    }

    fn drop(&mut self, rep: Resource<HostTrainTrackerDyn>) -> wasmtime::Result<()> {
        debug_assert!(rep.owned());
        let _train_tracker = self.table.delete(rep)?;
        Ok(())
    }
}

impl decthings::model::model_callbacks::Host for Host {}
