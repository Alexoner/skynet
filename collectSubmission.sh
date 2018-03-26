rm -f submission.zip
zip -r submission.zip . -x "*.git*" "*skynet/datasets*" "*.ipynb_checkpoints*" "*README.md" "*collectSubmission.sh" "*requirements.txt" ".env/*" "*.pyc" "*skynet/build/*"
