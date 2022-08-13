self: super: {
  galois = self.callPackage ./galois { };
  stim = self.callPackage ./stim { };
  qldpc = self.callPackage ./qldpc { };
  ldpc = self.callPackage ./ldpc { };
}