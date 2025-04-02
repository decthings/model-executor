use wasmtime::component::{Resource, ResourceTable};

pub type HostDataLoaderDyn = Box<dyn super::DataLoader>;

pub type HostWeightsProviderDyn = Box<dyn super::WeightsProvider>;

pub type HostWeightsLoaderDyn = Box<dyn super::WeightsLoader>;

pub type HostTrainTrackerDyn = Box<dyn super::TrainTracker>;

wasmtime::component::bindgen!({
    with: {
        "decthings:model/model-callbacks/data-loader": HostDataLoaderDyn,
        "decthings:model/model-callbacks/weights-provider": HostWeightsProviderDyn,
        "decthings:model/model-callbacks/weights-loader": HostWeightsLoaderDyn,
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

impl decthings::model::model_callbacks::HostWeightsProvider for Host {
    fn provide(
        &mut self,
        self_: Resource<HostWeightsProviderDyn>,
        data: Vec<(String, Vec<u8>)>,
    ) -> wasmtime::Result<()> {
        debug_assert!(!self_.owned());
        let weights_provider = self.table.get(&self_)?;
        weights_provider.provide(data);
        Ok(())
    }

    fn drop(&mut self, rep: Resource<HostWeightsProviderDyn>) -> wasmtime::Result<()> {
        debug_assert!(rep.owned());
        let _weights_provider = self.table.delete(rep)?;
        Ok(())
    }
}

impl decthings::model::model_callbacks::HostWeightsLoader for Host {
    fn read(&mut self, self_: Resource<HostWeightsLoaderDyn>) -> wasmtime::Result<Vec<u8>> {
        debug_assert!(!self_.owned());
        let weights_provider = self.table.get(&self_)?;
        Ok(weights_provider.read())
    }

    fn drop(&mut self, rep: Resource<HostWeightsLoaderDyn>) -> wasmtime::Result<()> {
        debug_assert!(rep.owned());
        let _weights_loader = self.table.delete(rep)?;
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
        self_: wasmtime::component::Resource<HostTrainTrackerDyn>,
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
