from neuralforecast.models import TimesNet, LSTM, GRU, KAN, VanillaTransformer
import torch
class CustomNHITS(TimesNet):
    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(params=self.parameters(), rho=0.75)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='min',factor=0.5, patience=3,
        )
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
            'monitor': 'valid_loss',
            'strict': True,
            'name': None,
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}


class CustomLSTM(LSTM):
    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(params=self.parameters(), rho=0.75)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='min',factor=0.5, patience=3,
        )
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
            'monitor': 'valid_loss',
            'strict': True,
            'name': None,
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}
    

class CustomGRU(GRU):
    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(params=self.parameters(), rho=0.75)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='min',factor=0.5, patience=3,
        )
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
            'monitor': 'valid_loss',
            'strict': True,
            'name': None,
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}
    
class CustomKAN(KAN):
    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(params=self.parameters(), rho=0.75)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='min',factor=0.5, patience=3,
        )
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
            'monitor': 'valid_loss',
            'strict': True,
            'name': None,
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}
    
class CustomVanillaTransformer(VanillaTransformer):
    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(params=self.parameters(),rho=0.75)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, mode='min',factor=0.5, patience=3,
        )
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
            'monitor': 'valid_loss',
            'strict': True,
            'name': None,
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}   