for j = 15:200   
im=imread([['figure'], num2str(sprintf('%04d',j)), '.png'],'png');
    [X,Ig]=rgb2ind(im,256);
    %M(i)=Ic;
    %X(:,:,1,i)=rgb2ind(Ic,256,'nodither');
    if j==2
        imwrite(X,Ig,'AnimationRH1.gif','gif','DelayTime',0.2,'LoopCount',1,'DisposalMethod','restorePrevious'); %g443800
    else
        imwrite(X,Ig,'AnimationRH1.gif','gif','DelayTime',0.2,'WriteMode','append'); %g443800
    end
    close all;
end