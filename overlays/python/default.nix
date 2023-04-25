(final: prev:
let
  pythonPackageOverlay = self: super: {
    qldpc = self.callPackage ./qldpc { };
    ldpc = self.callPackage ./ldpc { };
    stim = self.callPackage ./stim { };
    galois = self.callPackage ./galois { };
  };
in
{
  pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [ pythonPackageOverlay ];
})

