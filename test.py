    def start(self, dataset) -> Tuple[BaseKGE, str]:
        """Train selected model via the selected training strategy"""
        from ray.tune.integration.pytorch_lightning import TuneReportCallback

        print("------------------- Train -------------------")
        # (1) Perform K-fold CV
        if self.args.num_folds_for_cv >= 2:
            return self.k_fold_cross_validation(dataset)
        else:
            self.trainer: Union[TorchTrainer, TorchDDPTrainer, pl.Trainer]
            self.trainer = self.initialize_trainer(
                callbacks=get_callbacks(self.args), plugins=[]
            )
            if "pykeen" in self.args.model.lower():
                model, form_of_labelling = self.initialize_or_load_model(dataset)
            else:
                model, form_of_labelling = self.initialize_or_load_model()
            assert self.args.scoring_technique in [
                "KvsSample",
                "1vsAll",
                "KvsAll",
                "NegSample",
            ]
            train_loader = self.initialize_dataloader(
                self.initialize_dataset(dataset, form_of_labelling)
            )

        if isinstance(model, LitModule):
            # model.train_dataloaders.dataset.collate_fn = train_loader.dataset.collate_fn # ddp trainer needs this function
            self.trainer.fit(model,train_dataloaders=model.train_dataloaders)
            return model, form_of_labelling

        # hyparameter tune by ray
        # self.tune_mnist_asha(data_loader = train_loader)

        self.trainer.fit(model, train_dataloaders=train_loader)

        return model, form_of_labelling