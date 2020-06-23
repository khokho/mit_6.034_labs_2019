# MIT 6.034 Lab 7: Support Vector Machines
# Written by 6.034 staff

from svm_data import *
from functools import reduce


#### Part 1: Vector Math #######################################################

def dot_product(u, v):
    """Computes the dot product of two vectors u and v, each represented 
    as a tuple or list of coordinates. Assume the two vectors are the
    same length."""
    return sum(x[0]*x[1] for x in zip(u,v))

def norm(v):
    """Computes the norm (length) of a vector v, represented 
    as a tuple or list of coords."""
    return (sum(x*x for x in v))**(1/2)


#### Part 2: Using the SVM Boundary Equations ##################################

def positiveness(svm, point):
    """Computes the expression (w dot x + b) for the given Point x."""
    return dot_product(svm.w, point)+svm.b

def classify(svm, point):
    """Uses the given SVM to classify a Point. Assume that the point's true
    classification is unknown.
    Returns +1 or -1, or 0 if point is on boundary."""
    v = positiveness(svm,point)
    if v>0:
        return 1
    elif v<0:
        return -1
    else:
        return 0

def margin_width(svm):
    """Calculate margin width based on the current boundary."""
    return 2/norm(svm.w)

def check_gutter_constraint(svm):
    """Returns the set of training points that violate one or both conditions:
        * gutter constraint (positiveness == classification, for support vectors)
        * training points must not be between the gutters
    Assumes that the SVM has support vectors assigned."""
    ret = set()
    for p in svm.training_points:
        v = positiveness(svm, p)
        if abs(v)<1:
            ret.add(p)
        elif p.alpha > 0 and v != p.classification:
            ret.add(p)
    return ret
            


#### Part 3: Supportiveness ####################################################

def check_alpha_signs(svm):
    """Returns the set of training points that violate either condition:
        * all non-support-vector training points have alpha = 0
        * all support vectors have alpha > 0
    Assumes that the SVM has support vectors assigned, and that all training
    points have alpha values assigned."""
    nsv = set(svm.training_points)-set(svm.support_vectors)
    sv = set(svm.support_vectors)
    return set(p for p in nsv if p.alpha!=0) | set(p for p in sv if p.alpha<=0)


def check_alpha_equations(svm):
    """Returns True if both Lagrange-multiplier equations are satisfied,
    otherwise False. Assumes that the SVM has support vectors assigned, and
    that all training points have alpha values assigned."""
    return 0 == sum(p.alpha*p.classification for p in svm.training_points) \
        and svm.w==reduce(vector_add, [scalar_mult(p.alpha*p.classification, p) for p in svm.training_points])

#### Part 4: Evaluating Accuracy ###############################################

def misclassified_training_points(svm):
    """Returns the set of training points that are classified incorrectly
    using the current decision boundary."""
    return set(p for p in svm.training_points if p.classification != classify(svm, p))


#### Part 5: Training an SVM ###################################################

def update_svm_from_alphas(svm):
    """Given an SVM with training data and alpha values, use alpha values to
    update the SVM's support vectors, w, and b. Return the updated SVM."""
    sv = set(p for p in svm.training_points if p.alpha > 0)
    svm.support_vectors = list(sv)
    w = reduce(vector_add, [scalar_mult(p.alpha*p.classification, p) for p in svm.training_points])
    svp = set(p for p in sv if p.classification == 1)
    svn = set(p for p in sv if p.classification == -1)
    b = (max(-dot_product(w, p) for p in svp) \
            + min(-dot_product(w, p) for p in svn))/2
    svm.set_boundary(w,b)
    #print([positiveness(svm, p) for p in sv])
    return svm



#### Part 6: Multiple Choice ###################################################

ANSWER_1 = 11
ANSWER_2 = 6
ANSWER_3 = 3
ANSWER_4 = 2

ANSWER_5 = ['A', 'D']
ANSWER_6 = ['A', 'B', 'D']
ANSWER_7 = ['A', 'B', 'D']
ANSWER_8 = []
ANSWER_9 = ['A', 'B', 'D']
ANSWER_10 = ['A', 'B', 'D']

ANSWER_11 = False
ANSWER_12 = True
ANSWER_13 = False
ANSWER_14 = False
ANSWER_15 = False
ANSWER_16 = True

ANSWER_17 = [1, 3, 6, 8]
ANSWER_18 = [1, 2, 4, 5, 6, 7, 8]
ANSWER_19 = [1, 2, 4, 5, 6, 7, 8]

ANSWER_20 = 6


#### SURVEY ####################################################################

NAME = "Aleksandre Khokhiashvili"
COLLABORATORS = None
HOW_MANY_HOURS_THIS_LAB_TOOK = 3
WHAT_I_FOUND_INTERESTING = 'not the end'
WHAT_I_FOUND_BORING = 'end'
SUGGESTIONS = 'too much guesswork'
