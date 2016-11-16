function y_matrix = one_hot_encode(y_vector, n_classes)


y_matrix = zeros(size(y_vector, 1 ), n_classes);

% assuming class labels start from one
for i = 1:n_classes
    rows = y_vector == i;
    y_matrix(rows, i) = 1;
end