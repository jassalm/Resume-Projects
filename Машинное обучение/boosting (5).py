from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.tree import DecisionTreeRegressor

from typing import Optional

import matplotlib.pyplot as plt

from IPython.display import clear_output
from sklearn.preprocessing import KBinsDiscretizer

from scipy.sparse import csr_matrix, csc_matrix


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        early_stopping_rounds: int = None,
        subsample: float | int = 1.0,
        bagging_temperature: float | int = 1.0,
        bootstrap_type: str | None = 'Bernoulli',
        rsm: float | int = 1.0,
        quantization_type: str | None = None, 
        nbins: int = 255
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate

        self.history = defaultdict(list) # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean() # логистическая функция потреть ln(1 + exp(-M)) --> min                                                      
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)  # Исправьте формулу на правильную. 

        self.early_stopping_rounds: int = early_stopping_rounds

        self.bootstrap_type: str | None = bootstrap_type
        self.subsample: float | int = subsample
        self.bagging_temperature: float | int = bagging_temperature

        self.rsm: float | int = rsm
        self.features: list = list()
        self.quantization_type: str | None = quantization_type
        self.nbins: int = nbins

        self.feature_importances_ = None

    def partial_fit(self, X, y, old_predictions):
        bernulli_bp_msk = np.random.binomial(1, self.subsample, X.shape[0]).astype(bool)
        X_bp = X[bernulli_bp_msk]
        old_predictions_bp = old_predictions[bernulli_bp_msk]

        if self.bagging_temperature != 1:
            bayesian_weights = (-np.log(np.random.uniform(0, 1, X_bp.shape[0]))) ** 2
        else:
            bayesian_weights = None
        
        model = self.base_model_class(**self.base_model_params)
        model.fit(X_bp, old_predictions_bp, sample_weight=bayesian_weights)
        self.models.append(model)
        self.gammas.append(self.find_optimal_gamma(y, old_predictions, model.predict(X)))
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, plot=False, plot_process=False):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool 
        """
        if self.quantization_type is not None:
            X_train = self.binary_transform(X_train, method=self.quantization_type, n_bins=self.nbins)
            if X_val is not None:
                X_val = self.binary_transform(X_val, method=self.quantization_type, n_bins=self.nbins)
            
            
        train_predictions = np.zeros(y_train.shape[0])
        
        if y_val is not None:
            val_predictions = np.zeros(y_val.shape[0])
            
        previous_val_loss = np.inf
        previous_val_auc = -np.inf
        count = 0
        for i in range(self.n_estimators):
            
            train_loss = self.loss_fn(y_train, train_predictions)
            train_auc = roc_auc_score(y_train, train_predictions)

            self.history['train_auc'].append(train_auc)
            self.history['train_loss'].append(train_loss)

            if self.rsm != 1.0:
                #print('hi')
                feature_indices = np.random.choice(X_train.shape[1], size=int(self.rsm * X_train.shape[1]), replace=False)
            else:
                #print('noo', self.rsm)
                feature_indices = [i for i in range(X_train.shape[1])]
            X_train_sampled = X_train[:, feature_indices]
            self.features.append(feature_indices)
            
            residuals = -self.loss_derivative(y_train, train_predictions)         
            self.partial_fit(X_train_sampled, y_train, residuals)
            
            train_predictions += self.gammas[-1] * self.learning_rate * self.models[-1].predict(X_train_sampled)
    
            if y_val is not None: 
                
                val_loss = self.loss_fn(y_val, val_predictions)
                val_auc = roc_auc_score(y_val, val_predictions)
                self.history['val_auc'].append(val_auc)            
                self.history['val_loss'].append(val_loss)
                
                if val_loss < previous_val_loss and val_auc > previous_val_auc:
                    count = 0
                    previous_val_loss = val_loss
                    previous_val_auc = val_auc
                else:
                    previous_val_loss = val_loss
                    previous_val_auc = val_auc
                    count += 1
                    
                X_val_sampled = X_val[:, feature_indices]
                val_predictions += self.gammas[-1] * self.learning_rate * self.models[-1].predict(X_val_sampled)

            if self.early_stopping_rounds is not None:
                if count >= self.early_stopping_rounds:
                    break
                
            if plot_process:
                self.plot_history(self.history, range(i + 1), in_train=True)
                clear_output(True)
            
        if plot:
            self.plot_history(self.history, range(i + 1), in_train=True)

        self.compute_feature_importances(X_train.shape[1])

    def predict_proba(self, X):
        if self.quantization_type is not None:
             X = self.binary_transform(X, method=self.quantization_type, n_bins=self.nbins)      
        predictions = np.zeros(X.shape[0])
        for gamma, model, feats in zip(self.gammas, self.models, self.features):
            predictions += gamma * self.learning_rate * model.predict(X[:, feats])
        predictions = self.sigmoid(predictions)
        proba = np.zeros([X.shape[0], 2])
        proba[:, 0], proba[:, 1] = 1 - predictions, predictions
        return proba
        

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, X, y):
        return score(self, X, y)

    def binary_transform(self, X, method='uniform', n_bins=2):
        if method == 'uniform':
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        elif method == 'quantile':
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        else:
            raise ValueError("Method must be 'uniform' or 'quantile'")
        return csc_matrix(discretizer.fit_transform(X.toarray()))

    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def compute_feature_importances(self, number_of_feats):
        importances = np.zeros(number_of_feats)

        for model in self.models:
            importances += model.feature_importances_

        self.feature_importances_ = self.softmax(importances)
         
    def plot_history(self, X, y, in_train=False):
        """
        :param X: features array (any set)
        :param y: targets array (any set)
        """
        if in_train:
            plt.figure(figsize=(16, 9))
    
            plt.subplot(1, 2, 1)
            plt.plot(y, X['train_loss'], color='red', label='Train Loss')
            try:
                plt.plot(y, X['val_loss'], color='blue', label='Validation Loss')
            except:
                #print('No val')
                pass
            plt.legend(loc='lower right')
            
            plt.subplot(1, 2, 2)
            plt.plot(y, X['train_auc'], color='red', label='Train AUC')
            try:      
                plt.plot(y, X['val_auc'], color='blue', label='Validation AUC')
            except:
                #print('No val')
                pass
            plt.legend(loc='lower right')
    
            plt.show()
        else:
            predictions = np.zeros(y.shape[0])
            loss_auc = defaultdict(list)
    
            for gamma, model, feats in zip(self.gammas, self.models, self.features):
                
                predictions += gamma * self.learning_rate * model.predict(X[:, feats])
                loss = self.loss_fn(y, predictions)
                auc = roc_auc_score(y, predictions)
    
                loss_auc['auc'].append(auc)
                loss_auc['loss'].append(loss)
                
            plt.figure(figsize=(16, 9))
            
            plt.subplot(1, 2, 1)
            plt.plot(range(len(loss_auc['loss'])), loss_auc['loss'], color='red', label='Loss')
            plt.title('Loss')
            
            plt.subplot(1, 2, 2)
            plt.plot(range(len(loss_auc['auc'])), loss_auc['auc'], color='red', label='AUC')
            plt.title('AUC')

            plt.show()

def roc_pr_curves(y_test, preds):

    plt.figure(figsize=(16, 7))
    
    # PR-кривая
    plt.subplot(1, 2, 1)
    plt.title('PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    prec, recall, thresholds = precision_recall_curve(y_test, preds)
    prc_auc = auc(recall, prec)
    plt.plot(recall, prec, label=f'PRC_AUC: {prc_auc:.2f}')
    plt.legend(loc='lower right')
    
    # ROC-кривая
    plt.subplot(1, 2, 2) 
    plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], 'g--')
    
    fpr, tpr, _ = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUC: {roc_auc:.2f}, Gini: {roc_auc * 2 - 1:.2f}')
    
    plt.legend(loc='lower right')
    
    plt.tight_layout()        
