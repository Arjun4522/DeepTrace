{ pkgs ? import <nixpkgs> {} }:

let
  pythonPackages = pkgs.python313Packages;
in
pkgs.mkShell {
   buildInputs = with pkgs; [
     # Python with all required packages
     (pythonPackages.python.withPackages (ps: with ps; [
       numpy
       torch
       scapy
       faiss
       redis
       flask
       flask-cors
       flask-socketio
       python-socketio
       eventlet
       langchain
       pandas
       matplotlib
       seaborn
       requests
       # For development
       python-lsp-server
       black
       pylint
     ]))
     
     # System dependencies
     gcc
     stdenv.cc.cc.lib  # This provides libstdc++.so.6
     
     # Redis server
     redis
   ];
  
  # Ensure libstdc++ is available
  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
  
  shellHook = ''
    echo "DeepTrace development environment ready!"
  '';
}