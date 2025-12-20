function index = findInterval(value, array)

%--------------------------------------------------------------------------
% Find the interval containing a value inside an ordered array. Return the
% lower and upper indexes. If the value is exactly the same as one of the
% entries of the array, return the index of that entry. If the values is
% beyond the array values, return the lowest or highest index.
%--------------------------------------------------------------------------

if value <= array(1)
    index = 1;
elseif value >= array(end)
    index = length(array);
else
    [min_diff,index_closest] = min(abs(array - value));
    if min_diff == 0
        index = index_closest;
    else
        if array(index_closest) < value
            index = [index_closest, index_closest+1];
        else
            index = [index_closest-1, index_closest];
        end
    end
end
