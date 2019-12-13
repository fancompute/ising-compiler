from ising_compiler.gates_nx import IsingCircuit


class IsingALU(IsingCircuit):

    def HALF_ADDER(self, A, B, S = None, C = None):
        S_bit = self.XOR(A, B, out = S)
        C_bit = self.AND(A, B, out = C)
        return S_bit, C_bit

    def FULL_ADDER(self, A, B, Cin, S = None, Cout = None):
        aXORb = self.XOR(A, B)#, out = "A^B")
        aANDb = self.AND(A, B)#, out = "A&B")
        S_bit = self.XOR(aXORb, Cin, out = S)
        aXORbANDc = self.AND(aXORb, Cin)#, out = "(A^B)&C")
        Cout_bit = self.OR(aANDb, aXORbANDc, out = Cout)
        return S_bit, Cout_bit

    def FULL_ADDER_NAND(self, A, B, Cin, S = None, Cout = None):
        u1 = self.NAND(A, B, out = "u1")
        u2 = self.NAND(A, u1, out = "u2")
        u3 = self.NAND(u1, B, out = "u3")
        u4 = self.NAND(u2, u3, out = "u4")
        u5 = self.NAND(u4, Cin, out = "u5")
        u6 = self.NAND(u4, u5, out = "u6")
        u7 = self.NAND(u5, Cin, out = "u7")
        S_bit = self.NAND(u6, u7, out = S)
        Cout_bit = self.NAND(u5, u1, out = Cout)
        return S_bit, Cout_bit

    def RIPPLE_CARRY_ADDER(self, num_bits):
        assert num_bits > 1
        # Set up a half adder for first bit
        A0 = self.INPUT("A0")
        B0 = self.INPUT("B0")
        S, C = self.HALF_ADDER(A0, B0, S = "S0", C = "C1")
        # Continue with full adders to desired size
        S_bits = [S] # list of all sum bits
        for i in range(1, num_bits):
            A = self.INPUT("A" + str(i))
            B = self.INPUT("B" + str(i))
            S, C = self.FULL_ADDER(A, B, C, S = "S" + str(i), Cout = "C" + str(i + 1))
            S_bits.append(S)
        # Register sum bits and last carry as output
        for s in S_bits:
            self.OUTPUT(s)
        self.OUTPUT(C)
        # Return the sum bits
        return S_bits, C