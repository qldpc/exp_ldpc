self: super: {
  qldpc = self.callPackage ./qldpc { };
  ldpc = self.callPackage ./ldpc { };
  stim = self.callPackage ./stim { };
  galois = self.callPackage ./galois { };
}
