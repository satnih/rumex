function out = clipEdges(A, sz)
    out = A((sz+1):(end-sz), (sz+1):(end-sz), :);
end