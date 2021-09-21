#!/usr/bin/env bash

echo $1 | openssl aes-256-cbc -d -a -pass pass:somepassword
