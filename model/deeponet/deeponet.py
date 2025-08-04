import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Any
from tqdm import tqdm
import logging
import os
import dotenv
import json
from sklearn.model_selection import train_test_split
from datetime import datetime

from sciml.data.preprocessing import get_mu_xs_sol

dotenv.load_dotenv()

PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    logging.warning("PROJECT_ROOT not found in environment variables. Using current directory.")
    PROJECT_ROOT = os.getcwd()

VERBOSE_LOGGING = True   

log_dir = os.path.join(PROJECT_ROOT, "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"fno_training_{current_time}.log")

logging.basicConfig(
    level=logging.INFO if VERBOSE_LOGGING else logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler() if VERBOSE_LOGGING else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

def make_json_serializable(obj):
    if isinstance(obj, (tf.keras.Model, tf.keras.Sequential, tf.keras.layers.Layer)):
        return f"{obj.__class__.__name__}"
    elif isinstance(obj, tf.keras.optimizers.Optimizer):
        return f"{obj.__class__.__name__}(lr={obj.learning_rate.numpy() if hasattr(obj.learning_rate, 'numpy') else obj.learning_rate})"
    elif isinstance(obj, tf.keras.losses.Loss):
        return f"{obj.__class__.__name__}"
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj




class DeepONet(tf.keras.Model):
    ### DeepONet class ###
    def __init__(self, hyper_params: dict, regular_params: dict,folder_path:str): 
        """
        regular_params: dict
            - internal_model: tensorflow model for the internal basis-coefficient learning function, (input functions at grid points) R^d_p -> R^d_v (coefficients)
            - external_model: tensorflow model for the basis learning function, R^d_v -> R^d_v
            
            
        
        hyper_params: dict
            ### Model parameters ###
            - d_p: dimension of the encoder space for input function, number of grid points in the encoder
            - d_V: dimension of the decoder space for output function, number of basis functions to be learned
            
            ### Training parameters ###
            - learning_rate: learning rate for the optimizer, default is 0.001
            - optimizer: tensorflow optimizer for the model, default is Adam
            - n_epochs: number of epochs to train the model, default is 100
            - batch_size: batch size for the training, default is 32
            - verbose: verbosity for the training, default is 1
            - loss_function: tensorflow loss function for the model, default is MSE
            - device: device for inference and training the model on, default is 'cpu'
            
        Remarks : regular params have their own (hyper)parameters, which are not passed to the model, it is constructed in the build_model function
        """
        
        
        super().__init__()
        
        required_params = ["internal_model","external_model"]
        for param in required_params:
            if param not in regular_params:
                logger.error(f"Required parameter {param} not found in regular_params")
        
        
        self.params = {
            "hyper_params": hyper_params,
            "regular_params": regular_params
        }
        
        self.hyper_params = hyper_params
        self.regular_params = regular_params
        
        self.internal_model = self.regular_params["internal_model"]
        self.external_model = self.regular_params["external_model"]
        
        
        self.d_p = hyper_params["d_p"]
        self.d_V = hyper_params["d_V"]
        self.learning_rate = hyper_params["learning_rate"] if "learning_rate" in hyper_params else 0.001
        self.optimizer = hyper_params["optimizer"] if "optimizer" in hyper_params else tf.optimizers.Adam(self.learning_rate) # AdamW to be added after
        self.n_epochs = hyper_params["n_epochs"] if "n_epochs" in hyper_params else 100
        self.batch_size = hyper_params["batch_size"] if "batch_size" in hyper_params else 32
        self.verbose = hyper_params["verbose"] if "verbose" in hyper_params else 1
        self.loss_function = hyper_params["loss_function"] if "loss_function" in hyper_params else tf.losses.MeanSquaredError()
        self.device = hyper_params["device"] if "device" in hyper_params else 'cpu'
        self.folder_path = None  
    
        self.output_dim = hyper_params["output_dim"] if "output_dim" in hyper_params else None
        self.folder_path = folder_path
        
        logger.info(f"Model initialized with {self.n_epochs} epochs, {self.batch_size} batch size, {self.learning_rate} learning rate")
    
    
    @property
    def trainable_variables(self): # for user experience to get the trainable variables, doesn't required by tf
        return self.internal_model.trainable_variables + self.external_model.trainable_variables
        
    def predict(self, mu: tf.Tensor, x: tf.Tensor):
        with tf.device(self.device):
            #  branch network
            coefficients = self.internal_model(mu)  # [batch, d_V]
            batch_size = tf.shape(x)[0]
            # trunk network, basis evaluation
            # if x is already in the format [batch, n_points, dim_coords], treat it directly
            if len(x.shape) == 3:
                # flatten to treat each point as a separate input, batch differenciation is unuseful
                n_points = tf.shape(x)[1]
                x_flat = tf.reshape(x, [-1, x.shape[-1]])  # [batch*n_points, dim_coords]
                basis_flat = self.external_model(x_flat)  # [batch*n_points, d_V], to be fed to the external               
                basis_evaluation = tf.reshape(basis_flat, [batch_size, n_points, -1])  # [batch, n_points, d_V]
                output = tf.einsum('bi,bji->bj', coefficients, basis_evaluation)  # tensor contraction [batch, n_points]
                return output
            else:
                raise ValueError(f"Format de x incorrect. Attendu [batch, n_points, dim_coords], reçu {x.shape}")
    
    # in case you want to modify those models but not the other
    def set_internal_model(self,internal_model:tf.keras.Model): # for user experience to tune something
        self.internal_model = internal_model
        
    def set_external_model(self,external_model:tf.keras.Model): # for user experience to tune something
        self.external_model = external_model
        
        
        
        
        
    
    
    ### Data loading ### Be careful with the data format, we can have various sensor points for parameters : for instance a specified mu function can require to get many more points to compute the exact solution
    def get_data(self,folder_path:str) -> tuple[tf.Tensor,tf.Tensor]: # typing is important
        
        true_path = os.path.join(PROJECT_ROOT,folder_path)
        self.folder_path = true_path
        
        try: # error handling because it's critical
            
            mu_files = [np.load(os.path.join(true_path,f"mu/mu_{i}.npy")) for i in tqdm(range(min(len(os.listdir(os.path.join(true_path,"mu"))),100)), desc="Loading mu data")]
            x_files = [np.load(os.path.join(true_path,f"xs/xs_{i}.npy")) for i in tqdm(range(min(len(os.listdir(os.path.join(true_path,"xs"))),100)), desc="Loading x data")]
            sol_files = [np.load(os.path.join(true_path,f"sol/sol_{i}.npy")) for i in tqdm(range(min(len(os.listdir(os.path.join(true_path,"sol"))),100)), desc="Loading y data")]
            
            
            mus = tf.convert_to_tensor(mu_files, dtype=tf.float32)
            mus = tf.reshape(mus, [tf.shape(mus)[0], -1])
            
            xs = tf.convert_to_tensor(x_files, dtype=tf.float32) 
            if len(xs.shape) > 2:  
                n_samples = xs.shape[0]
                xs = tf.reshape(xs, [n_samples, -1, xs.shape[-1]])  # reshape en [batch, n_points, dim_coords]
        
            sol = tf.convert_to_tensor(sol_files, dtype=tf.float32)
            sol = tf.reshape(sol, [tf.shape(sol)[0], -1])
        except:
            logger.error(f"Data not found in {true_path}")
            raise ValueError(f"Data not found in {true_path}")
        
        return mus, xs, sol
    
    def get_data_given(self,folder_path:str,type:float,training:bool=True)->tuple[tf.Tensor,tf.Tensor,tf.Tensor]:
        mus,xs,sol = get_mu_xs_sol(folder_path,type,training)
        return mus,xs,sol
    
    # mandatory methods to be implemented for keras
    def call(self,mu:tf.Tensor,x:tf.Tensor)->tf.Tensor:
        return self.predict(mu,x)
    
    
    def compile(self): # apparently mandatory to compile the model
        self.optimizer = self.hyper_params["optimizer"] if "optimizer" in self.hyper_params else tf.optimizers.Adam(self.learning_rate)
        self.loss_function = self.hyper_params["loss_function"] if "loss_function" in self.hyper_params else tf.losses.MeanSquaredError()
    
    def build(self): # apparently also mandatory to build the model for tensorflow
        self.internal_model.build(input_shape=(None,self.d_p))
        self.external_model.build(input_shape=(None,self.d_V))
    
    
    
    ### managing model training methods ###
    def fit(self,device:str='GPU',inputs=None,sol=None,given=True,type=0.2,training=True)->np.ndarray:
        if given:
            mus,xs, sol = self.get_data_given(self.folder_path,type,training)
        else:
            mus,xs, sol = self.get_data(self.folder_path)
        loss_history_train = []
        loss_history_test = []
        

        
        with tf.device(device):
            dataset = tf.data.Dataset.from_tensor_slices((mus,xs,sol)) # batching the data with batch size
            train_dataset,test_dataset = tf.keras.utils.split_dataset(dataset,left_size=0.8,right_size=0.2,seed=42)
            train_dataset = train_dataset.batch(self.batch_size) # batching method from tensorflow
            test_dataset = test_dataset.batch(self.batch_size) # batching method from tensorflow
       
            
            for epoch in tqdm(range(self.n_epochs),desc="Training progress"):
                mean_loss = 0
                for batch in train_dataset:
                    loss = self.train_step(batch)
                    mean_loss += loss
                loss_history_train.append(float(mean_loss/len(train_dataset)))
                    
                mean_loss = 0
                for batch in test_dataset:
                    batchloss = self.test_step(batch)
                    mean_loss += batchloss
                loss_history_test.append(float(mean_loss/len(test_dataset)))
                
                date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                
                
                logger.info(f"Epoch {epoch+1}/{self.n_epochs}")
                logger.info(f"Training Loss: {loss_history_train[-1]:.6f}")
                logger.info(f"Test Loss: {loss_history_test[-1]:.6f}")
                if max(loss_history_train[-1],loss_history_test[-1]) < 0.001:
                    break
                
        return loss_history_train,loss_history_test
    
    def test_step(self,batch:tuple[tf.Tensor,tf.Tensor,tf.Tensor])->tf.Tensor:
        mu,x,sol = batch
        y_pred = self.predict(mu,x)
        loss = self.loss_function(y_pred,sol)
        return loss
            
        
    def train_step(self, batch: tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Étape d'entraînement simplifiée - minimum de code
        """
        mu, x, sol = batch
        
        with tf.GradientTape() as tape:
            # Prédiction
            y_pred = self.predict(mu, x)
            
            # Calcul direct de la perte
            loss = self.loss_function(y_pred, sol)
        
        # Backpropagation
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss
            
            
            
    ## managing model saving methods        
    def save(self,save_path:str):  # error handling because it's also critical out there, we save a tensorflow model as a keras file
        if not os.path.exists(save_path):
            os.makedirs(save_path,exist_ok=True)
        
        try:
            tf.saved_model.save(self,save_path)
        except:
            logger.error(f"Failed to save model in {save_path}")
            raise ValueError(f"Failed to save model in {save_path}")
        


    def load_weights(self,save_path:str): # just loading some other weights if we want to compare, but not the entire model
        if not os.path.exists(save_path):
            logger.error(f"Weights not found in {save_path}")
            raise ValueError(f"Weights not found in {save_path}")
        
        self.load_weights(save_path)
        
    def save_weights(self,save_path:str): 
        if not os.path.exists(save_path):
            os.makedirs(save_path,exist_ok=True)
        
        self.save_weights(save_path)
