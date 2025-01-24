{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
      pythonEnv = pkgs.python3.withPackages (ps: with ps; [
        (ps.torch.override {
          cudaSupport = true;
          cudaPackages = pkgs.cudaPackages;
        })
        pandas
        librosa
        tqdm
        transformers
      ]);
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          pythonEnv
          cudatoolkit
          cudnn
          libsndfile
        ];

        # Required for CUDA to find libraries
        shellHook = ''
          export LD_LIBRARY_PATH=${pkgs.cudatoolkit}/lib:${pkgs.cudnn}/lib:$LD_LIBRARY_PATH
          export CUDA_PATH=${pkgs.cudatoolkit}
        '';
      };
    };
}
