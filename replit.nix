{ pkgs }: {
  deps = [
    pkgs.python311
    pkgs.nodejs_20
    pkgs.nodePackages.npm
    pkgs.ffmpeg-full
    pkgs.libGL
    pkgs.glib
  ];
}
