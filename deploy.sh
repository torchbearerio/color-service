#!/bin/bash

# Build zip
rm -f build.zip
zip -r build.zip environment.yml colorservice Dockerfile .ebextensions

# Deploy to EB
eb deploy
