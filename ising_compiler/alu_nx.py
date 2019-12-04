from ising_compiler.gates_nx import IsingCircuitGraph


class IsingCircuitModules(IsingCircuitGraph):


    def HALF_ADDER(self, A, B, S=None, C=None):
        S_bit = self.XOR(A, B, out=S)
        C_bit = self.AND(A, B, out=C)
        return S_bit, C_bit

    def FULL_ADDER(self, A, B, Cin, S = None, Cout = None):
        aXORb = self.XOR(A, B, out="A^B")
        aANDb = self.AND(A, B, out="A&B")
        S_bit = self.XOR(aXORb, Cin, out = S)
        aXORbANDc = self.AND(aXORb, Cin, out = "(A^B)&c")
        Cout_bit = self.OR(aANDb, aXORbANDc, out = Cout)
        return S_bit, Cout_bit
