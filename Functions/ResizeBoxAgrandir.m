function bbox = ResizeBoxAgrandir(W,H,result)
bbox(1)=(W*result(1))/224;
bbox(3)=(W*result(3))/224;
bbox(2)=(H*result(2))/224;
bbox(4)=(H*result(4))/224;
end