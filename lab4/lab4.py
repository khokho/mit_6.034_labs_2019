# MIT 6.034 Lab 4: Constraint Satisfaction Problems
# Written by 6.034 staff

from constraint_api import *
from test_problems import get_pokemon_problem


#### Part 1: Warmup ############################################################

def is_ok(csp, x, y, xval, yval):
    for cs in csp.constraints_between(x,y):
        if not cs.check(xval, yval):
            return False
    return True

def has_empty_domains(csp) :
    """Returns True if the problem has one or more empty domains, otherwise False"""
    return len(min(csp.domains.values(), key=len))==0

def check_all_constraints(csp) :
    """Return False if the problem's assigned values violate some constraint,
    otherwise True"""
    for x in csp.assignments:
        xval = csp.get_assignment(x)
        for y in csp.get_neighbors(x):
            yval = csp.get_assignment(y)
            if yval is not None:
                for cs in csp.constraints_between(x,y):
                    if not cs.check(xval, yval):
                        return False
    return True



#### Part 2: Depth-First Constraint Solver #####################################

def solve_constraint_dfs(problem) :
    """
    Solves the problem using depth-first search.  Returns a tuple containing:
    1. the solution (a dictionary mapping variables to assigned values)
    2. the number of extensions made (the number of problems popped off the agenda).
    If no solution was found, return None as the first element of the tuple.
    """
    agenda = [problem]
    cnt = 0
    while len(agenda)>0:
        cur = agenda[0]
        agenda = agenda[1:]
        cnt+=1
        #print(cur)
        if has_empty_domains(cur) or not check_all_constraints(cur):
            continue
        #print(cur)
        ass = cur.pop_next_unassigned_var()
        if ass is None:
            return cur.assignments, cnt
        add = []
        for val in cur.get_domain(ass):
            add.append(cur.copy().set_assignment(ass, val))
        agenda[:0] = add
    return None, cnt

# QUESTION 1: How many extensions does it take to solve the Pokemon problem
#    with DFS?

# Hint: Use get_pokemon_problem() to get a new copy of the Pokemon problem
#    each time you want to solve it with a different search method.

print(solve_constraint_dfs(get_pokemon_problem()))

ANSWER_1 = 20


#### Part 3: Forward Checking ##################################################


def eliminate_from_neighbors(csp, x) :
    """
    Eliminates incompatible values from var's neighbors' domains, modifying
    the original csp.  Returns an alphabetically sorted list of the neighboring
    variables whose domains were reduced, with each variable appearing at most
    once.  If no domains were reduced, returns empty list.
    If a domain is reduced to size 0, quits immediately and returns None.
    """
    modified = []
    for y in csp.get_neighbors(x):
        start_domain = list(csp.get_domain(y))
        for yval in start_domain:
            ok = False
            for xval in csp.get_domain(x):
                if is_ok(csp, x, y, xval, yval):
                    ok = True
            if not ok:
                csp.eliminate(y, yval)
        if len(start_domain)!=len(csp.get_domain(y)):
            modified.append(y)
        if len(csp.get_domain(y))==0:
            return None
    return modified


# Because names give us power over things (you're free to use this alias)
forward_check = eliminate_from_neighbors

def solve_constraint_forward_checking(problem) :
    """
    Solves the problem using depth-first search with forward checking.
    Same return type as solve_constraint_dfs.
    """
    agenda = [problem]
    cnt = 0
    while len(agenda)>0:
        cur = agenda[0]
        agenda = agenda[1:]
        cnt+=1
        #print(cur)
        if has_empty_domains(cur) or not check_all_constraints(cur):
            continue
        #print(cur)
        ass = cur.pop_next_unassigned_var()
        if ass is None:
            return cur.assignments, cnt
        add = []
        for val in cur.get_domain(ass):
            new_problem = cur.copy().set_assignment(ass, val)
            forward_check(new_problem, ass)
            add.append(new_problem)
        agenda[:0] = add
    return None, cnt



# QUESTION 2: How many extensions does it take to solve the Pokemon problem
#    with DFS and forward checking?


print(solve_constraint_forward_checking(get_pokemon_problem()))

ANSWER_2 = 9


#### Part 4: Domain Reduction ##################################################

def domain_reduction(csp, queue=None) :
    """
    Uses constraints to reduce domains, propagating the domain reduction
    to all neighbors whose domains are reduced during the process.
    If queue is None, initializes propagation queue by adding all variables in
    their default order. 
    Returns a list of all variables that were dequeued, in the order they
    were removed from the queue.  Variables may appear in the list multiple times.
    If a domain is reduced to size 0, quits immediately and returns None.
    This function modifies the original csp.
    """
    if queue is None:
        queue = csp.get_all_variables()
    deq = []
    while len(queue) > 0:
        cur = queue[0]
        deq.append(cur)
        queue = queue[1:]
        reduced = forward_check(csp, cur)
        if reduced is None:
            return None
        queue.extend(reduced)
    return deq



# QUESTION 3: How many extensions does it take to solve the Pokemon problem
#    with DFS (no forward checking) if you do domain reduction before solving it?
reduced = get_pokemon_problem().copy()
domain_reduction(reduced)
print(solve_constraint_dfs(reduced))
ANSWER_3 = 6



def solve_constraint_propagate_reduced_domains(problem) :
    """
    Solves the problem using depth-first search with forward checking and
    propagation through all reduced domains.  Same return type as
    solve_constraint_dfs.
    """
    agenda = [problem]
    cnt = 0
    while len(agenda)>0:
        cur = agenda[0]
        agenda = agenda[1:]
        cnt+=1
        #print(cur)
        if has_empty_domains(cur) or not check_all_constraints(cur):
            continue
        #print(cur)
        ass = cur.pop_next_unassigned_var()
        if ass is None:
            return cur.assignments, cnt
        add = []
        for val in cur.get_domain(ass):
            new_problem = cur.copy().set_assignment(ass, val)
            domain_reduction(new_problem, [ass])
            add.append(new_problem)
        agenda[:0] = add
    return None, cnt


# QUESTION 4: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through reduced domains?

print(solve_constraint_propagate_reduced_domains( get_pokemon_problem()))
ANSWER_4 = 7


#### Part 5A: Generic Domain Reduction #########################################

def propagate(enqueue_condition_fn, csp, queue=None) :
    """
    Uses constraints to reduce domains, modifying the original csp.
    Uses enqueue_condition_fn to determine whether to enqueue a variable whose
    domain has been reduced. Same return type as domain_reduction.
    """
    if queue is None:
        queue = csp.get_all_variables()
    deq = []
    while len(queue) > 0:
        cur = queue[0]
        deq.append(cur)
        queue = queue[1:]
        reduced = forward_check(csp, cur)
        if reduced is None:
            return None
        reduced = [x for x in reduced if enqueue_condition_fn(csp, x)]
        queue.extend(reduced)
    return deq

def condition_domain_reduction(csp, var) :
    """Returns True if var should be enqueued under the all-reduced-domains
    condition, otherwise False"""
    return True

def condition_singleton(csp, var) :
    """Returns True if var should be enqueued under the singleton-domains
    condition, otherwise False"""
    return len(csp.get_domain(var))==1

def condition_forward_checking(csp, var) :
    """Returns True if var should be enqueued under the forward-checking
    condition, otherwise False"""
    return False


#### Part 5B: Generic Constraint Solver ########################################

def solve_constraint_generic(problem, enqueue_condition=None) :
    """
    Solves the problem, calling propagate with the specified enqueue
    condition (a function). If enqueue_condition is None, uses DFS only.
    Same return type as solve_constraint_dfs.
    """
    agenda = [problem]
    cnt = 0
    while len(agenda)>0:
        cur = agenda[0]
        agenda = agenda[1:]
        cnt+=1
        #print(cur)
        if has_empty_domains(cur) or not check_all_constraints(cur):
            continue
        #print(cur)
        ass = cur.pop_next_unassigned_var()
        if ass is None:
            return cur.assignments, cnt
        add = []
        for val in cur.get_domain(ass):
            new_problem = cur.copy().set_assignment(ass, val)
            if enqueue_condition is not None:
                propagate(enqueue_condition, new_problem, [ass])
            add.append(new_problem)
        agenda[:0] = add
    return None, cnt

# QUESTION 5: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through singleton domains? (Don't
#    use domain reduction before solving it.)


print(solve_constraint_generic(get_pokemon_problem(), condition_singleton))

ANSWER_5 = 8


#### Part 6: Defining Custom Constraints #######################################

def constraint_adjacent(m, n) :
    """Returns True if m and n are adjacent, otherwise False.
    Assume m and n are ints."""
    return abs(n-m)==1

def constraint_not_adjacent(m, n) :
    """Returns True if m and n are NOT adjacent, otherwise False.
    Assume m and n are ints."""
    return not abs(n-m)==1

def not_same(m, n) :
    """Returns True if m and n are NOT same, otherwise False.
    Assume m and n are ints."""
    return n!=m

def all_different(variables) :
    """Returns a list of constraints, with one difference constraint between
    each pair of variables."""
    ret = []
    for i in range(len(variables)):
        for j in range(i+1, len(variables)):
            x = variables[i]
            y = variables[j]
            if x==y:
                continue
            ret.append(Constraint(x, y, not_same))
    return ret



#### SURVEY ####################################################################

NAME = "Aleksandre Khokhiashvili"
COLLABORATORS = None
HOW_MANY_HOURS_THIS_LAB_TOOK = 2
WHAT_I_FOUND_INTERESTING = "mostly everythin"
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
