from collections import defaultdict
import sys

def eval_train(pred_filename):
  '''
  Evaluates the score of a training prediction file.
  Arguments:
    pred_filename: The filename of a prediction file.
  Returns:
    (domain_scores, overall_score)
    domain_scores: List of accuracies for each domain, in order.
    overall_score: The mean (across domains) of the accuracy
      (within each domain), i.e. the mean of domain_scores.
  '''
  # Parse training annotations
  domain_total = defaultdict(lambda:0)
  domains = []
  classes = []
  with open('train_annos.txt', 'r') as anno_file:
    for line in anno_file:
      line = line.split(',')
      domain = int(line[2])
      domains.append(domain)
      classes.append(int(line[3]))
      domain_total[domain] += 1
  num_domains = max(domains)
  # Parse predictions
  with open(pred_filename, 'r') as pred_file:
    preds = [int(x.strip()) for x in pred_file.readlines()]
  correct = map(lambda (x,y):x==y, zip(classes, preds))
  domain_correct = defaultdict(lambda:0)
  for domain, good_pred in zip(domains, correct):
    domain_correct[domain] += good_pred
  domain_scores = [0] * num_domains
  for domain in domain_total:
    domain_scores[domain-1] = float(domain_correct[domain])/domain_total[domain]
  overall_score = sum(domain_scores) / num_domains
  return (domain_scores, overall_score)

if __name__ == '__main__':
  # Run the thing
  if len(sys.argv) != 2:
    print 'Usage: eval_train.py pred_filename'
    sys.exit()
  domain_scores, overall_score = eval_train(sys.argv[1])
  print 'Domain scores: %s' % domain_scores
  print 'Overall score: %g' % overall_score
