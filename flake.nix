{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    #flake-utils.url = "github:numtide/flake-utils/master";
  };

  outputs = { self, nixpkgs }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
    };
  in {

    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        cudatoolkit
      ];
      shellHook = ''
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/run/opengl-driver/lib
     '';
    };

  };
}
