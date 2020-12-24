rm -r ../data/eval/test/*.xml
rm -r ../data/eval/prettify/*.xml
rm -r ../data/eval/guess/*.xml

python3 Bert.py trigger eval
python3 Evaluation.py ../data/eval/gold/ ../data/eval/guess/

