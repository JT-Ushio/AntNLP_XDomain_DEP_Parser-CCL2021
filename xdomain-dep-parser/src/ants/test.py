
class B:
    def cando(self, a):
        print(a.val)


class A:

    def __init__(self, val, b):
        b.cando(self)
        self.val = val

    def p(self, b):
        b.cando(self)

def main():
    
    b = B()
    a = A(12, b)



if __name__ == '__main__':
    main()


