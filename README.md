# py2r
    usage: import * from numpy_to_r
    Transfer some functions from R to numpy&scipy 
    窝真的不习惯用python写以前R做的东西，于是写了这么个东西。。。
    List of functions:
        c(a[,b,...])
        rep(x,size)
        seq(a,b,delta)
        matrix(arr[,ncol,nrow])
        array(arr,dim)
        rbind(a,b)
        cbind(a,b)
        t(mat)
        solve(mat)
        which(a,b,str)
            this is different from original function 'which' in R,
            usage: which(c(1,2,3),c(2,3,1),'<=') -> c(0,1)
        which_and(a[,b,...]) & which_or(a[,b,...])
            these two functions are used to complete the usage of original function 'which' in R,
            usage: 
                  which_and(c(1,2,3),c(2,3,4)) -> c(2,3)
                  which_or(c(1,2,3),c(2,3,4)) -> c(1,2,3,4)
        apply_all(arr,fun)
            apply fun to all elements in arr
        apply_row(mat,fun)
            apply fun to all rows in mat
        apply_col(mat,fun)
            apply fun to all cols in mat
        read_csv(filename[,sep='\t',header=False])
        write_csv(filename[,sep='\t',header=False])
        norm family
        pois family
        exp family
        t family
        f family
        gamma family
        chisq family
        beta family
        runif *(not unif family)
        lm(y,x)
            x could be matrix
