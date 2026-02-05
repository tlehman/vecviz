{
  description = "VecViz - 3D Embedding Visualization";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          numpy
          scikit-learn
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            # Go
            pkgs.go

            # Python with dependencies
            pythonEnv

            # SQLite (for sqlite-vec CGO)
            pkgs.sqlite

            # Build tools
            pkgs.gcc
            pkgs.pkg-config
          ];

          shellHook = ''
            echo "VecViz development environment"
            echo "  Go: $(go version)"
            echo "  Python: $(python3 --version)"
            echo ""
            echo "Commands:"
            echo "  go build -o vecviz .   # Build the server"
            echo "  ./vecviz               # Run the server"
          '';

          # CGO settings for sqlite-vec
          CGO_ENABLED = "1";
        };
      }
    );
}
