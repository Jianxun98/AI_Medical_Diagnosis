import numpy as np
import matplotlib.pyplot as plt
from utils import *
np.random.seed(42) ## This line helps the reproducibility of the values

y = np.loadtxt('data/y_true_balanced_set.txt')
pred = np.loadtxt('data/pred_balanced_set.txt')

print(f"Proportion of positives: {np.sum(y==1)/len(y)}")
print(f"Proportion of negatives: {np.sum(y==0)/len(y)}")

pred_threshold = pred >= 0.5

TN = np.sum((y == 0) & (pred_threshold == 0))
TP = np.sum((y == 1) & (pred_threshold == 1))
FN = np.sum((y == 1) & (pred_threshold == 0))
FP = np.sum((y == 0) & (pred_threshold == 1))

print(f"Number of examples: {len(y)}\nNumber of True Negatives: {TN}\nNumber of True Positives: {TP}\nNumber of False Negatives: {FN}\nNumber of False Positives: {FP}")

# Let's compute the FPR, TPR, and Precision for the data examples we loaded. 
# We will use two functions from the utils library
# called get_fpr and get_tpr. You can take a look at them using help(get_tpr) and help(get_fpr)!

FPR = get_fpr(y,pred)
TPR = get_tpr(y,pred)
precision = TP/(TP+FP)

print(f"FPR (1 - specificity): {FPR:.4f}\nTPR (sensitivity or recall): {TPR:.4f}\nPrecision: {precision:.4f}\nSpecificity: {(1-FPR):.4f}")

# Run this cell 
tpr = [] # In this list there will be appended the tpr values for each threshold
fpr = [] # In this list there will be appended the fpr values for each threshold
for th in np.arange(0,1,0.01):
    tpr.append(get_tpr(y,pred,th))
    fpr.append(get_fpr(y,pred,th))


# Now let's use matplotlib to plot the function
plt.plot(fpr,tpr)
plt.plot(0,1,'ro')
plt.legend(['Estimator'])
plt.annotate(f'  ({0}, {1})',xy = (0,1))
plt.xlabel('False Positive Rate (1 - specificity)')
plt.ylabel('True Positive Rate (sensitivity)')
plt.title('ROC Curve for the estimator')

# Run this cell
def random_guesser(x):
    # For every point x, it returns a random point between 0 and 1. 
    # Note that it do not use any information about the point x!
    return round(np.random.rand(),2) 

random_predictions = [random_guesser(x) for x in y] 
print(random_predictions[:20])


# Run this cell 
random_tpr = [] 
random_fpr = [] 
for th in np.arange(0,1,0.01):
    random_tpr.append(get_tpr(y,random_predictions,th))
    random_fpr.append(get_fpr(y,random_predictions,th))

# Now let's use matplotlib to plot the function
plt.plot(random_fpr,random_tpr)
plt.plot(fpr,tpr)
plt.legend(['Random Guesser','Actual Predictor'])
plt.plot(0,1,'ro')
plt.annotate(f'  ({0}, {1})',xy = (0,1))
plt.xlabel('False Positive Rate (1 - specificity)')
plt.ylabel('True Positive Rate (sensitivity)')
plt.title('ROC Curves for the estimators')


# Now let's use matplotlib to plot the function
plt.plot(random_fpr,random_tpr)
plt.plot(fpr,tpr)
plt.plot(np.arange(0,1,0.01),np.arange(0,1,0.01), 'r--')
plt.legend(['Random Guesser','Actual Predictor'])
plt.plot(0,1,'ro')
plt.annotate(f'  ({0}, {1})',xy = (0,1))
plt.xlabel('False Positive Rate (1 - specificity)')
plt.ylabel('True Positive Rate (sensitivity)')
plt.title('ROC Curves for the estimators')

y_2 = np.loadtxt('data/y_true_unbalanced_set.txt')
pred_2 = np.loadtxt('data/pred_unbalanced_set.txt')


print(f"Proportion of positives: {np.sum(y_2==1)/len(y_2)}")
print(f"Proportion of negatives: {np.sum(y_2==0)/len(y_2)}")

pred_2_threshold = pred_2 >= 0.5

TN = np.sum((y_2 == 0) & (pred_2_threshold == 0))
TP = np.sum((y_2 == 1) & (pred_2_threshold == 1))
FN = np.sum((y_2 == 1) & (pred_2_threshold == 0))
FP = np.sum((y_2 == 0) & (pred_2_threshold == 1))

print(f"Number of examples: {len(y_2)}\nNumber of True Negatives: {TN}\nNumber of True Positives: {TP}\nNumber of False Negatives: {FN}\nNumber of False Positives: {FP}")

# Cell 21
FPR = get_fpr(y_2,pred_2)
TPR = get_tpr(y_2,pred_2)
precision = TP/(TP+FP)

print(f"FPR (1 - specificity): {FPR:.4f}\nTPR (sensitivity or recall): {TPR:.4f}\nPrecision: {precision:.4f}\nSpecificity: {(1-FPR):.4f}")


tpr_2 = [] 
fpr_2 = [] 
for th in np.arange(0,1,0.01):
    tpr_2.append(get_tpr(y_2,pred_2,th))
    fpr_2.append(get_fpr(y_2,pred_2,th))

# Now let's use matplotlib to plot the function
plt.plot(fpr_2,tpr_2)
plt.plot(np.arange(0,1,0.01),np.arange(0,1,0.01), 'r--')
plt.legend(['Estimator'])
plt.plot(0,1,'ro')
plt.annotate(f'  ({0}, {1})',xy = (0,1))
plt.xlabel('False Positive Rate (1 - specificity)')
plt.ylabel('True Positive Rate (sensitivity)')
plt.title('ROC Curves for the estimators')


y_2 = np.loadtxt('data/y_true_unbalanced_set.txt')
pred_2 = np.loadtxt('data/pred_unbalanced_set.txt')

print(f"Proportion of positives: {np.sum(y_2==1)/len(y_2)}")
print(f"Proportion of negatives: {np.sum(y_2==0)/len(y_2)}")

print(f"Number of examples: {len(y_2)}\nNumber of True Negatives: {TN}\nNumber of True Positives: {TP}\nNumber of False Negatives: {FN}\nNumber of False Positives: {FP}")


# Cell 21
FPR = get_fpr(y_2,pred_2)
TPR = get_tpr(y_2,pred_2)
precision = TP/(TP+FP)

print(f"FPR (1 - specificity): {FPR:.4f}\nTPR (sensitivity or recall): {TPR:.4f}\nPrecision: {precision:.4f}\nSpecificity: {(1-FPR):.4f}")


tpr_2 = [] 
fpr_2 = [] 
for th in np.arange(0,1,0.01):
    tpr_2.append(get_tpr(y_2,pred_2,th))
    fpr_2.append(get_fpr(y_2,pred_2,th))

# Now let's use matplotlib to plot the function
plt.plot(fpr_2,tpr_2)
plt.plot(np.arange(0,1,0.01),np.arange(0,1,0.01), 'r--')
plt.legend(['Estimator'])
plt.plot(0,1,'ro')
plt.annotate(f'  ({0}, {1})',xy = (0,1))
plt.xlabel('False Positive Rate (1 - specificity)')
plt.ylabel('True Positive Rate (sensitivity)')
plt.title('ROC Curves for the estimators')


def euclidean_distance(x,y):
    """
    Compute the euclidean distance between two vectors in R2
    Args:
        x,y (np.array or list or tuple): R2 vectors
    Returns:
        d (float): The euclidean distance between x and y
    """
    d = ((x[0] - y[0])**2 + (x[1] - y[1])**2)**0.5
    return d


# We will build a list with the following structure: [(threshold, associated point on ROC Curve)]
examples_and_thresholds = []
for th in np.arange(0,1,0.01):
    examples_and_thresholds.append(((th,(get_fpr(y_2,pred_2,th),get_tpr(y_2,pred_2,th)))))
examples_and_thresholds[0]

threshold_and_distances = []
for th,point in examples_and_thresholds:
    threshold_and_distances.append((th,euclidean_distance((0,1),point),point))
    
threshold_and_distances.sort(key = lambda x: x[1])

print(f"The chosen threshold therefore is {threshold_and_distances[0][0]}")

# Now let's use matplotlib to plot the function
plt.plot(fpr_2,tpr_2)
plt.plot(np.arange(0,1,0.01),np.arange(0,1,0.01), 'r--')
plt.legend(['Estimator'])
plt.plot(0,1,'ro')
plt.plot(*threshold_and_distances[0][2], 'ro')
plt.annotate(f'  ({0}, {1})',xy = (0,1))
plt.annotate(f'  ({threshold_and_distances[0][2][0]:.2f}, {threshold_and_distances[0][2][1]:.2f})',xy = threshold_and_distances[0][2])
plt.xlabel('False Positive Rate (1 - specificity)')
plt.ylabel('True Positive Rate (sensitivity)')
plt.title('ROC Curves for the estimators')


pred_2_threshold = pred_2 >= 0.08

TN = np.sum((y_2 == 0) & (pred_2_threshold == 0))
TP = np.sum((y_2 == 1) & (pred_2_threshold == 1))
FN = np.sum((y_2 == 1) & (pred_2_threshold == 0))
FP = np.sum((y_2 == 0) & (pred_2_threshold == 1))

print(f"Number of examples: {len(y_2)}\nNumber of True Negatives: {TN}\nNumber of True Positives: {TP}\nNumber of False Negatives: {FN}\nNumber of False Positives: {FP}")

FPR = get_fpr(y_2,pred_2,0.08)
TPR = get_tpr(y_2,pred_2,0.08)
precision = TP/(TP+FP)

print(f"FPR (1 - specificity): {FPR:.4f}\nTPR (sensitivity or recall): {TPR:.4f}\nPrecision: {precision:.4f}\nSpecificity: {(1-FPR):.4f}")













