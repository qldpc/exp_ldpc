self: super: {
  stim = self.callPackage ./stim { };
  qldpc = self.callPackage ./qldpc { };
  ldpc = self.callPackage ./ldpc { };
}