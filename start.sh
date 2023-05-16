docker run -d --rm --name seed-reverser --runtime=nvidia --gpus all -v /opt/endcity:/data $(docker build -q .)
