import re
import sys

stmt_tok_patterns = [
    (r'\(', 'LPAREN'),
    (r'\)', 'RPAREN'),
    (r'[A-Za-z_][A-Za-z0-9_]*', 'ATOM'),
    (r'\\bot', 'BOT'),
    (r'\->', 'IMPL'),
    (r'\!', 'NOT'),
    (r'/\\', 'AND'),
    (r'\\/', 'OR'),
    (r'\|-', 'DERIVES'),
    (r'\,', 'SEP'),
    (r'\[[^\]]+\]', 'EXPL')
]

token_stmt_regex = '|'.join(f'({pattern})' for pattern, token_type in stmt_tok_patterns)

def lex_stmt(expression):
    tokens = []
    while expression:
        match = re.match(token_stmt_regex, expression)
        if match:
            for i in range(1, len(match.groups()) + 1):
                if match.group(i):
                    token_value = match.group(i)
                    token_type = stmt_tok_patterns[i - 1][1]
                    tokens.append((token_type, token_value))
                    expression = expression[len(token_value):].lstrip()
                    break
        # If unable to match
        else:
          print("Unable to tokenize expression")
          break
    return tokens

# Test the lexer
# expression = "[or-el 4, 5-6, 7-9] p /\ r"
# expression = "q -> r, p, q |- p \/ q -> p \/ r"
# tokens = lex_stmt(expression)
# print(tokens)
# for token_type, token_value in tokens:
#     print(f"Token Type: {token_type}, Token Value: {token_value}")

# EXPRESSION TREE DECLARATION

class TreeNode:
  def __init__(self, val, left, right):
    self.val = val
    self.left = left
    self.right = right

# For testing
def inorder_traversal(root):
  if root is not None:
      inorder_traversal(root.left)
      print(root.val, end=' ')
      inorder_traversal(root.right)

def parse_stmt(s, i):
  if (len(s) == 0):
    return (None,1)
  c = s[i]
  i += 1
  if (c[0]=='ATOM'):
      return (TreeNode(c[1], None, None), i)

  if (c[0] == 'BOT'):
     return (TreeNode(c[0], None, None), i)

  if (c[0] == 'NOT'):
    # (exp, j) = parse_stmt(s, i)
    assert s[i][0] == 'LPAREN', 'syntax error'
    i += 1
    if (s[i+1][0] == 'RPAREN'):
      return (TreeNode('NOT', None, TreeNode(s[i][0], None, None)), i+2)
    (left, j) = parse_stmt(s, i)
    i = j
    op = s[i][0]
    i += 1
    (right, j) = parse_stmt(s, i)
    i = j
    c = s[i]
    assert c[0] == 'RPAREN', 'syntax error'
    i+= 1
    # return (, i)
    return (TreeNode('NOT', None, TreeNode(op, left, right)), j)

  if c[0] == 'LPAREN':
      if (s[i][0] == 'NOT'):
        i += 1
        (exp, j) = parse_stmt(s, i)
        i = j
        c = s[i]
        assert c[0] == 'RPAREN', 'syntax error'
        i += 1
        return (TreeNode('NOT', None, exp), i)
      else:
        (left, j) = parse_stmt(s, i)
        i = j
        op = s[i][0]
        i += 1
        (right, j) = parse_stmt(s, i)
        i = j
        c = s[i]
        assert c[0] == 'RPAREN', 'syntax error'
        i+= 1
        return (TreeNode(op, left, right), i)

def parse(s):
  return parse_stmt(s,0)[0]

# expr = "((!q) -> r)"
# toks = lex_stmt(expr)
# print(toks)
# # node = parse_stmt(toks,0)[0]
# node = parse(toks)
# inorder_traversal(node)

expl_tok_patterns = [
    (r'\[', 'IGNORE'),
    (r'\]', 'IGNORE'),
    (r'[a-zA-z]+[-A-Za-z0-9]+', 'RULE'),
    # (r'[a-zA-z]+\-[a-zA-z0-9]+', 'RULE'),
    (r'[0-9]+', 'LINENUM'),
    (r'\,', 'SEP'),
    (r'-', 'RANGETO'),
]

token_expl_regex = '|'.join(f'({pattern})' for pattern, token_type in expl_tok_patterns)

def lex_expl_stmt(expression):
    tokens = []
    while expression:
        match = re.match(token_expl_regex, expression)
        if match:
            for i in range(1, len(match.groups()) + 1):
                if match.group(i):
                    token_value = match.group(i)
                    token_type = expl_tok_patterns[i - 1][1]
                    tokens.append((token_type, token_value))
                    expression = expression[len(token_value):].lstrip()
                    break
        # If unable to match
        else:
          print("Unable to tokenize expression")
          break
    return tokens

# Test the lexer
# expression = "[or-el 4, 5-6, 7-9] p /\ r"
# expl = "[or-el 4, 5-6, 7-9]"
# toks_expl = lex_expl_stmt(expl)
# print(toks_expl)
# for token_type, token_value in toks_expl:
#     print(f"Token Type: {token_type}, Token Value: {token_value}")

## Decode each explanation

def decode_expl_toks(toks):
  opcode = None
  lines = []
  ranges = []
  i = 0
  assert toks[i][0] == 'IGNORE', 'syntax error'
  i += 1
  curr_range = []

  while (toks[i][0] != 'IGNORE'):
    c = toks[i]
    if (c[0] == 'RULE'):
      opcode = c[1]
      i += 1
    elif (c[0] == 'LINENUM'):
      if (len(curr_range) == 0):
        if (toks[i+1][0] == 'SEP' or toks[i+1][0] == 'IGNORE'):
          lines.append(int(c[1])-1)
          i+= 1
        else:
            curr_range.append(int(c[1])-1)
            i += 2
      else:
        curr_range.append(int(c[1])-1)
        ranges.append(tuple(curr_range))
        curr_range = []
        i += 1
    else:
      i += 1
  return (opcode, lines, ranges)

# decode_expl_toks(toks_expl)

## Check a line of the proof

def check_expr_trees(t1, t2):
  if (t1 == None and t2 == None):
    return True
  elif (t1 != None and t2 != None):
    if (t1.val != t2.val):
      return False
    else:
      return check_expr_trees(t1.left, t2.left) and check_expr_trees(t1.right, t2.right)
  else:
    return False

# Function to check if line is in scope
def chk_line_in_scope(ct, idx, scope, scope_start_end_map):
  # ct = current statement idx
  # idx = statement to be checked idx
  # check if the closest scope of idx contains the scope of ct
  for s in scope[ct]:
    if (s == scope[idx][0]):
      return True
  return False

def chk_line_in_par_scope(ct, idx, scope, scope_start_end_map):
  # ct = current statement idx
  # idx = statement to be checked idx
  # check if the parent scope of idx is the scope of ct
  f = scope[idx]
  if (len(f) > 1):
    x = f[1]
    for s in scope[ct]:
      if (s == x):
        return True
    return False
  return False

def chk_proof_line(statement, exprs, premises, opcode, lines, ranges,
                   ct, scope, scope_start_end_map):
  ## Check based on operand
  if (opcode == 'premise'):
    # inorder_traversal(statement)
    # print("\n")
    for pr in premises:
      # inorder_traversal(pr)
      # print("\n")
      if check_expr_trees(pr, statement):
        return True
    return False

  elif (opcode == 'assumption'):
    return True

  elif (opcode == 'copy'):
    # Allow copy from a parent
    chk0 = chk_line_in_scope(ct, lines[0], scope, scope_start_end_map)
    # check_line_in_scope_or_parent(lines[0], scope, scope_start_end_map)
    return chk0 and check_expr_trees(exprs[lines[0]], statement)

  elif (opcode == 'mp'):
    j = lines[0] # a
    i = lines[1] # a -> b
    chk1 = (exprs[i].val == 'IMPL')
    chk2 = check_expr_trees(exprs[i].left, exprs[j])
    chk3 = check_expr_trees(exprs[i].right, statement)
    chk4 = chk_line_in_scope(ct, j, scope, scope_start_end_map)
    chk5 = chk_line_in_scope(ct, i, scope, scope_start_end_map)
    # print(chk1, chk2, chk3)
    return chk1 and chk2 and chk3 and chk4 and chk5

  elif (opcode == 'mt'):
    i = lines[0] # a -> b
    j = lines[1] # neg b
    chk1 = (exprs[i].val == 'IMPL')
    chk2 = exprs[j].val == 'NOT' and check_expr_trees(exprs[i].right, exprs[j].right)
    chk3 = statement.val == 'NOT' and check_expr_trees(exprs[i].left, statement.right)
    chk4 = chk_line_in_scope(ct, i, scope, scope_start_end_map)
    chk5 = chk_line_in_scope(ct, j, scope, scope_start_end_map)
    # print(chk1, chk2, chk3, chk4, chk5)
    return chk1 and chk2 and chk3 and chk4 and chk5

  elif (opcode == 'and-in'):
    i = lines[0] # a
    j = lines[1] # b
    chk1 = (statement.val == 'AND')
    chk2 = check_expr_trees(statement.left, exprs[i])
    chk3 = check_expr_trees(statement.right, exprs[j])
    chk4 = chk_line_in_scope(ct, i, scope, scope_start_end_map)
    chk5 = chk_line_in_scope(ct, j, scope, scope_start_end_map)
    return chk1 and chk2 and chk3 and chk4 and chk5

  elif (opcode == 'and-e1'):
    i = lines[0] # a ^ b
    chk1 = (exprs[i].val == 'AND')
    chk2 = check_expr_trees(exprs[i].left, statement)
    # chk3 = check_expr_trees(exprs[i].left, statement)
    chk3 = chk_line_in_scope(ct, i, scope, scope_start_end_map)
    return chk1 and chk2 and chk3

  elif (opcode == 'and-e2'):
    i = lines[0] # a ^ b
    chk1 = (exprs[i].val == 'AND')
    chk2 = check_expr_trees(exprs[i].right, statement)
    # chk3 = check_expr_trees(exprs[i].left, statement)
    chk3 = chk_line_in_scope(ct, i, scope, scope_start_end_map)
    return chk1 and chk2

  elif (opcode == 'or-in1'):
    i = lines[0] # a
    chk1 = check_expr_trees(statement.left, exprs[i])
    chk2 = chk_line_in_scope(ct, i, scope, scope_start_end_map)
    return chk1 and chk2

  elif (opcode == 'or-in2'):
    i = lines[0] # a
    chk1 = check_expr_trees(statement.right, exprs[i])
    chk2 = chk_line_in_scope(ct, i, scope, scope_start_end_map)
    return chk1 and chk2

  elif (opcode == 'or-el'):
    i = lines[0] # a
    j,k = ranges[0]
    l,m = ranges[1]

    chk0 = (exprs[i].val == "OR")
    phi = exprs[i].left
    psi = exprs[i].right

    chk1 = check_expr_trees(phi, exprs[j])
    chk2 = check_expr_trees(psi, exprs[l])
    # inorder_traversal(exprs[k])
    # print("\n")
    # inorder_traversal(exprs[m])
    # print("\n")
    # inorder_traversal(statement)
    # print("\n")
    chk3 = check_expr_trees(exprs[k], exprs[m])
    chk4 = check_expr_trees(exprs[k], statement)
    # print(chk0, chk1, chk2, chk3, chk4)

    # Scope checking
    chk5 = chk_line_in_scope(ct, i, scope, scope_start_end_map)
    chk6 = chk_line_in_par_scope(ct, j, scope, scope_start_end_map)
    chk7 = chk_line_in_par_scope(ct, l, scope, scope_start_end_map)

    return chk0 and chk1 and chk2 and chk3 and chk4 and chk5 and chk6 and chk7

  elif (opcode == 'impl-in'):
    i, j = ranges[0]
    chk1 = (statement.val == 'IMPL')
    chk2 = check_expr_trees(statement.left, exprs[i])
    chk3 = check_expr_trees(statement.right, exprs[j])
    # print(chk1,chk2,chk3)
    chk4 = chk_line_in_par_scope(ct, i, scope, scope_start_end_map)
    chk5 = (i in scope_start_end_map) and (j == scope_start_end_map[i])
    # print(i,j, scope_start_end_map)
    # print(chk5)
    # inorder_traversal(exprs[i])
    # inorder_traversal(statement.left)
    # print("\n")
    # inorder_traversal(exprs[j])
    # inorder_traversal(statement.right)
    # print("\n")
    # print(chk1,chk2,chk3, chk4)
    return chk1 and chk2 and chk3 and chk4 and chk5

  elif (opcode == 'neg-in'):
    i, j = ranges[0]
    chk1 = (statement.val == 'NOT')
    chk2 = check_expr_trees(exprs[i], statement.right)
    chk3 = (exprs[j].val == 'BOT')
    chk4 = chk_line_in_par_scope(ct, i, scope, scope_start_end_map)
    return chk1 and chk2 and chk3 and chk4

  elif (opcode == 'neg-el'):
    i = lines[0]
    j = lines[1]
    chk1 = (exprs[j].val == 'NOT') and check_expr_trees(exprs[j].right, exprs[i])
    chk2 = (statement.val == 'BOT')
    chk3 = chk_line_in_scope(ct, i, scope, scope_start_end_map)
    chk4 = chk_line_in_scope(ct, j, scope, scope_start_end_map)
    return chk1 and chk2 and chk3 and chk4

  elif (opcode == 'bot-el'):
    i = lines[0]
    chk1 = (exprs[i].val == 'BOT')
    chk2 = chk_line_in_scope(ct, i, scope, scope_start_end_map)
    return chk1 and chk2

  elif (opcode == 'dneg-in'):
    i = lines[0]
    chk1 = (statement.val == 'NOT')
    chk2 = (statement.right != None and statement.right.val == 'NOT')
    chk3 = check_expr_trees(statement.right.right, exprs[i])
    chk4 = chk_line_in_scope(ct, i, scope, scope_start_end_map)
    return chk1 and chk2 and chk3 and chk4

  elif (opcode == 'dneg-el'):
    i = lines[0]
    chk1 = (exprs[i].val == 'NOT')
    chk2 = (exprs[i].right != None and exprs[i].right.val == 'NOT')
    chk3 = check_expr_trees(exprs[i].right.right, statement)
    chk4 = chk_line_in_scope(ct, i, scope, scope_start_end_map)
    return chk1 and chk2 and chk3 and chk4

  elif (opcode == 'pbc'):
    i, j = ranges[0]
    chk1 = (exprs[i].val == 'NOT')
    chk2 = check_expr_trees(exprs[i].right, statement)
    chk3 = (exprs[j].val == 'BOT')
    chk4 = chk_line_in_par_scope(ct, i, scope, scope_start_end_map)
    return chk1 and chk2 and chk3 and chk4

  elif (opcode == 'lem'):
    chk1 = (statement.val == 'OR')
    chk2 = (statement.right != None and statement.right.val == 'NOT')
    chk3 = check_expr_trees(statement.left, statement.right.right)
    # print(chk1,chk2,chk3)
    return chk1 and chk2 and chk3

  else:
    print("Unknown opcode!", opcode)
    return False

  return True

## SCOPE CHECKER

def get_lines_from_proof_file(pf_file):
  proof_file = open(pf_file, 'r')
  lines = proof_file.readlines()
  return lines

def get_scope_info(lines):
    # Stores the id of the closest binding scope for each line
    scope_map = {}
    scope_start_end_map = {}
    ct = 2

    m = ct
    while (m < len(lines)):
      scope_map[m] = list([])
      m += 1

    while(ct < len(lines)):
      # Parse the line
      toks = lex_stmt(lines[ct])

      expl_toks = lex_expl_stmt(toks[0][1])
      opcode, e_lines, e_ranges = decode_expl_toks(expl_toks)

      # Check to detect a scope
      close_scope_ops = ['neg-in', 'pbc', 'impl-in']
      if (opcode in close_scope_ops):
        i, j = e_ranges[0]
        # print(i,j)
        if (i in scope_start_end_map):
          scope_start_end_map[i] = max(j,scope_start_end_map[i])
        else:
          scope_start_end_map[i] = j
        for z in range(i, j+1):
          # print(z)
          scope_map[z].append(i)

      elif (opcode == 'or-el'):
        i, j = e_ranges[0]
        k, l = e_ranges[1]
        if (i in scope_start_end_map):
          scope_start_end_map[i] = max(j,scope_start_end_map[i])
        else:
          scope_start_end_map[i] = j
        if (k in scope_start_end_map):
          scope_start_end_map[k] = max(l,scope_start_end_map[k])
        else:
          scope_start_end_map[k] = l
        # scope_start_end_map[k] = l
        # print(i,j,k,l)
        for z in range(i, j+1):
          scope_map[z].append(i)
        for z in range(k, l+1):
          scope_map[z].append(k)
      ct += 1

    m = 2
    while(m < len(lines)):
      scope_map[m].append(2)
      m+=1

    # At the exit of the functions, each line now has the info
    # About the starting lines of ALL the scopes they are contained in
    return (scope_map, scope_start_end_map)

## NATURAL DEDUCTION PROOF CHECKER

## Main Function to Check Proof
def check_proof(proof_filename):
  ## Get lines from the input file
  lines = get_lines_from_proof_file(proof_filename)
  scope_map, scope_start_end_map = get_scope_info(lines)
  ct = 0

  # Premises and target
  premises = []
  target = None

  # Array to store the available expressions at each scope level
  # maps[0] stores the map at the topmost scope level
  scopes = []
  top_level_scope = {}
  scopes.append(top_level_scope)

  ## Lex the first line and get the premises and the proposition
  toks = lex_stmt(lines[ct])
  ct += 2

  i = 0
  premise = []
  # Get the premises
  while(True):
    if(toks[i][0] != 'SEP' and toks[i][0] != 'DERIVES'):
      premise.append(toks[i])
      i+=1
    else:
      if(toks[i][0] == 'SEP'):
        premises.append(parse(premise))
        premise = []
        i += 1
      else:
        # print(premise)
        premises.append(parse(premise))
        i+=1
        break
  # Set the target
  target = parse(toks[i:])

  while(ct < len(lines)):
    # For each line in the proof
    # First see the proof rule being applied and the line numbers
    # Then parse the rest of the line
    # Finally, conclude by seeing the line numbers and the proof rule
    # whether this line of the proof is correct or not
    # If correct, push it into the current scope
    toks = lex_stmt(lines[ct])
    # print(toks)

    # Process the explanation here
    expl_toks = lex_expl_stmt(toks[0][1])
    # print(expl_toks)
    opcode, e_lines, e_ranges = decode_expl_toks(expl_toks)

    # Check if we need to start a new scope
    if (opcode == 'assumption'):
      # Check if some start of scope has been detected here
      if ct not in scope_start_end_map:
        # print(scope_start_end_map)
        return False

    # Get scope of current line
    curr_line_scope = scope_map[ct]

    # if (opcode == 'assumption'):
    #   new_scope = scopes[-1].copy()
    #   scopes.append(new_scope)

    # Parse and convert the rest of the statement
    scopes[-1][ct] = parse(toks[1:])
    # inorder_traversal(scopes[-1][ct])

    # inorder_traversal(scopes[-1][ct])
    # print("\n---")

    # Check the proof line
    correct = chk_proof_line(scopes[-1][ct], scopes[-1], premises, \
                            opcode, e_lines, e_ranges, ct, \
                             scope_map, scope_start_end_map)

    if (not correct):
      return False
    else:
      print("Line ", ct+1, " correct")

    # Go to next line
    ct += 1

  # Cleanup:
  if check_expr_trees(scopes[-1][ct-1], target):
    return True
  return False

def checkProof(filename):
  res = check_proof(filename)
  if (res):
    return "correct"
  else:
    return "incorrect"

# Main Function
if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("Usage: python script.py <filename>")
    sys.exit(1)
  filename = sys.argv[1]
  res = checkProof(filename)
  print(res)