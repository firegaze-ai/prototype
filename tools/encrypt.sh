#!/usr/bin/env bash

echo $1 | openssl aes-256-cbc -a -salt -pass pass:somepassword
