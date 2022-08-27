self: super: {
  qldpc = self.callPackage ./qldpc { };
  ldpc = self.callPackage ./ldpc { };
}