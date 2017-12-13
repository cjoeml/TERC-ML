#!/bin/bash
shuf -zen8165 ./* | xargs -0 mv -t train
