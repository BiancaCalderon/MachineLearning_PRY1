import unittest
import torch
from Regresion_No_Lineal.models import RegressionModel
from Regresion_No_Lineal.dataset import SinDataset
from Regresion_No_Lineal.train import train_model
from Regresion_No_Lineal.test import test_model

#Regresión no lineal begins
class TestRegressionModel(unittest.TestCase):
    def test_model_training(self):
        # Test if the model can be trained without errors
        try:
            train_model(num_epochs=10, target_loss=0.1)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Model training failed with exception: {e}")

    def test_model_prediction(self):
        # Test if the model can make predictions
        model = RegressionModel()
        dataset = SinDataset(num_samples=100)
        x_test = dataset.x
        y_test = dataset.y

        with torch.no_grad():
            predictions = model(torch.tensor(x_test, dtype=torch.float32))
        
        self.assertEqual(predictions.shape, torch.tensor(y_test, dtype=torch.float32).shape)

    def test_model_saving_loading(self):
        # Test if the model can be saved and loaded
        model = RegressionModel()
        torch.save(model.state_dict(), 'test_model.pth')
        model.load_state_dict(torch.load('test_model.pth'))
        self.assertTrue(True)

##Regresión no lineal ends

if __name__ == "__main__":
    unittest.main()