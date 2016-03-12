function [domain_scores, overall_score] = eval_train(pred_filename)
% Evaluates the score of a training prediction file.
% INPUT:
%   pred_filename: The filename of a prediction file.
% OUTPUT:
%   domain_scores: Matrix of accuracies for each domain, in order.
%   overall_score: The mean (across domains) of the accuracy
%     (within each domain), i.e. the mean of domain_scores.
train_data = load('train_annos.mat');
num_domains = max([train_data.annotations.domain_index]);
domain_correct = zeros(1, num_domains);
domain_total = zeros(1, num_domains);
preds = csvread(pred_filename);
for i = 1:numel(train_data.annotations)
  domain = train_data.annotations(i).domain_index;
  domain_total(domain) = domain_total(domain) + 1;
  domain_correct(domain) = domain_correct(domain) + (preds(i) == train_data.annotations(i).class);
end
domain_scores = domain_correct ./ domain_total;
overall_score = mean(domain_scores);
