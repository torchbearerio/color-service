#!/bin/bash

# Build zip
zip build environment.yml saliencyservice/* Dockerfile

# Deploy to EB
eb deploy
