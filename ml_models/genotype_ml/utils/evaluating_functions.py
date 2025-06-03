#!/usr/bin/env python3

### Functions for model evaluation ###
#%%
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import pickle
import xgboost as xgb
import scipy
import sys


from utils.data_loading import _print

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluates the model's performance based on accuracy, classification report, and confusion matrix.
    
    Parameters:
    y_true (array-like): True labels or ground truth.
    y_pred (array-like): Predicted labels from the model.
    
    Returns:
    dict: A dictionary containing accuracy, classification report, and confusion matrix.
    """
    if isinstance(model, xgb.Booster) or isinstance(model, xgb.XGBModel):
        # Train predictions
        y_train_pred = model.predict(xgb.DMatrix(X_train))
        # Test predictions
        y_pred = model.predict(xgb.DMatrix(X_test))
    else: # MultinomialLogisticRegression
        # Train predictions
        _, y_train_pred = model.predict(torch.tensor(X_train.toarray(), dtype=torch.float32))
        # Test predictions
        _, y_pred = model.predict(torch.tensor(X_test.toarray(), dtype=torch.float32))

    train_cm = confusion_matrix(y_train, y_train_pred)  
    test_cm = confusion_matrix(y_test, y_pred)
    
    # Calculate accuracy
    train_acc = balanced_accuracy_score(y_train, y_train_pred)
    accuracy = balanced_accuracy_score(y_test, y_pred)
    
    # Generate classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Create test confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=test_cm)
    
    # Plot the confusion matrix
    disp.plot(cmap=plt.cm.Blues)
    
    # Get the current figure object
    plot_obj = plt.gcf()
    
    return accuracy, train_acc, class_report, train_cm, test_cm, plot_obj

def metrics_txt(results_dir, model_name, acc, train_acc, string_mets, best_model, train_cm, test_cm):
    with open(results_dir + f'{model_name}_metrics.txt', 'w') as file:
        file.write(f'Model name: {model_name}\n')
        file.write(f'Model balanced accuracy: {acc}\n')
        file.write(f'Model balanced train accuracy: {train_acc}\n')
        file.write('\nModel report:\n')
        for key, value in string_mets.items():
            file.write(f'{key}: {value}\n')
        file.write('\nModel parameters:\n')
        file.write(str(best_model.get_params()))
        file.write('\nTraining confusion matrix:\n')
        file.write(str(train_cm))
        file.write('\nTraining confusion matrix:\n')
        file.write(str(test_cm))

def save_results(results_dir, model_name, model, X_train, X_test, y_train, y_test, train_loss_histories, val_loss_histories):

    # Evaluate model
    accuracy, train_acc, class_report, train_cm, test_cm, plot_obj = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Save results
    metrics_txt(results_dir, model_name, accuracy, train_acc, class_report, model, train_cm, test_cm)
    plot_obj.savefig(results_dir + f'{model_name}_test_confmatx.png')
    plt.close(plot_obj)  # Close the confusion matrix plot

    # Save learning curves
    learning_curves = model.plot_loss(train_loss_histories, val_loss_histories)
    learning_curves.savefig(results_dir + f'{model_name}_learning.png')
    plt.close()  # Close the learning curves plot

    # Save model
    with open(results_dir + f'{model_name}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    _print("Saved results")