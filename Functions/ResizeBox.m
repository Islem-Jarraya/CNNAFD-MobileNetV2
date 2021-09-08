function bbox = ResizeBox(W,H,box)
WPourcentage=22400/W;
HPourcentage=22400/H;
bbox=[];
bbox(1)=(WPourcentage*box(1))/100;
bbox(3)=(WPourcentage*box(3))/100;
bbox(2)=(HPourcentage*box(2))/100;
bbox(4)=(HPourcentage*box(4))/100;
end