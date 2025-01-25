from operator import add, sub, mul, truediv

class Solver(object):
    
 #solver using infix expressions. Expects the input string to have
   # a space between each token, for example: ( 5 + 3 ) / 2.
    
    def __init__(self):
        self.op = {'+': add, '-': sub, '*': mul, '/': truediv}

    def to_postfix(self, s):
        opstack = []
        postfix = ''
        tokens = s.split()
        for t in tokens:
            if t in ['*', '/']:
                while opstack and opstack[-1] in ['*', '/']:
                    postfix += opstack.pop() + ' '
                opstack.append(t)
            elif t in ['+', '-']:
                while opstack and opstack[-1] != '(':
                    postfix += opstack.pop() + ' '
                opstack.append(t)
            elif t == '(':
                opstack.append(t)
            elif t == ')':
                while opstack[-1] != '(':
                    postfix += opstack.pop() + ' '
                opstack.pop()
            else:
                postfix += t + ' '
        while opstack:
            postfix += opstack.pop() + ' '
        return postfix

    def _evaluate(self, s):
        opstack = []
        tokens = s.split()
        for t in tokens:
            if t not in self.op:
                opstack.append(float(t))
            else:
                n1 = opstack.pop()
                n2 = opstack.pop()
                opstack.append(self.op[t](n2, n1))
        if len(opstack) != 1:
            raise Exception
        return opstack.pop()

    def evaluate(self, expression):
        try:
            result = self._evaluate(self.to_postfix(expression))
            return result
        except Exception:
            return f'wrong or not able to evaluate expression.'
