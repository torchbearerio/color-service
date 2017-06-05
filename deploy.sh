#!/bin/bash

# Build zip
zip -r build.zip environment.yml colorservice Dockerfile

# Deploy to EB
eb deploy
