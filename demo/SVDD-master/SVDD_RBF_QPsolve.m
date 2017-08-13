function [misclassifieds_array] = SVDD_RBF_QPsolve(patterns_filename, C_grid, testing_filename)
        %SVDD_train.dat is opened here as patternsfile
        patternsfile=fopen(patterns_filename, 'r');

        %The first two numbers from the file are read: total number of patterns, dimension
        total_patterns=fscanf(patternsfile, '%d', 1);
        dimension_patterns=fscanf(patternsfile, '%d', 1);
        %exercise to count how many of the patterns are typical, and not anomalous
        %(37) in Fast SVDD paper is based on only typical patterns

        pattern_point=zeros(dimension_patterns, 1);
        pattern_typical_count=0;
        for i=1:1:total_patterns
        for j=1:1:dimension_patterns
            pattern_point(j, 1)=fscanf(patternsfile, '%f', 1);
        end

        pattern_type=fscanf(patternsfile, '%d', 1);
        
        if (pattern_type==1)
            pattern_typical_count=pattern_typical_count+1;
        end
        end

        fprintf(1, '\nPattern dimension d: %d typical patterns: %d', dimension_patterns, pattern_typical_count);
        
        %closing and opening file again to begin reading points into data structures
        fclose(patternsfile);

        %preparing the data structures to be fed into the QP-solver quapro
        true_misclassifieds=0.0;
        X=zeros(pattern_typical_count, dimension_patterns);
        Q=zeros(pattern_typical_count, pattern_typical_count);
        p=zeros(pattern_typical_count, 1);
        C=ones(1, pattern_typical_count);
        b=ones(1, 1);
        ci=zeros(pattern_typical_count, 1);
        cs=ones(pattern_typical_count, 1);
        
        j=1;

        patternsfile=fopen(patterns_filename, 'r');
        total_patterns=fscanf(patternsfile, '%d', 1);
        dimension_patterns=fscanf(patternsfile, '%d', 1);
        for i=1:1:total_patterns
        for k=1:1:dimension_patterns
            pattern_point(k, 1)=fscanf(patternsfile, '%f', 1);
        end

        pattern_type=fscanf(patternsfile, '%d', 1);

        %points are read into the X data structure if found to be typical
        if (pattern_type==1)
            X(j, :)=pattern_point(:, 1)';
            p(j, 1)=0.0;
            j=j+1;
        end
        end
        
        fclose(patternsfile);
        

        percentile_array=zeros(1, pattern_typical_count*pattern_typical_count);

        for i=1:1:pattern_typical_count
            for j=1:1:pattern_typical_count
                norm_2=norm(X(i,:)-X(j,:))^2;
                percentile_array(1, (i-1)*pattern_typical_count+j)= norm_2;
            end
        end
        %percentile_values=prctile(percentile_array, [70, 80, 85, 90, 95]);
        percentile_values=prctile(percentile_array, [100,100,100,100,100]);
        JJ_limit=5;
        misclassifieds_array=zeros(C_grid, JJ_limit, dimension_patterns+3);
        
        for II=1:1:C_grid
        for JJ=1:1:JJ_limit;

        C_value=(2.0)^(-2+II);
        gamma_value=1.0/(percentile_values(1, JJ));
        fprintf(1, '\n(C:%0.2f gamma:%0.2f) ', C_value, gamma_value);
        
        for i=1:1:pattern_typical_count
            for j=1:1:pattern_typical_count
                norm_2=percentile_array(1, (i-1)*pattern_typical_count+j);
                Q(i, j)=2.0*exp(-gamma_value*norm_2);
            end
            cs(i, 1)=C_value*cs(i, 1);
        end    
        
        %Q is Q-matrix in QP, p is vector of constants in the objective function.
        %C is constraint matrix (in our case a vector), b is RHS of constraints (our case a scalar), ci-cs are lower-upper limits
        options=optimset('Algorithm', 'interior-point-convex', 'Display', 'off', 'TolFun', 1e-16);
        %options=optimset('Algorithm', 'interior-point-convex', 'Display', 'off');
        [QP_betas, fopt, lagr, OP_struct]=quadprog(Q, p, [], [], C, b, ci, cs, [], options);
        %counting the number of Unbounded Support Vectors (UBSV)
        if (lagr > -2)
        support_vector_count=0;
        UBSVs=0;
        QP_betas_sum=0.0;
        
        for i=1:1:pattern_typical_count
            if (QP_betas(i, 1) > 1e-5)
               support_vector_count=support_vector_count+1;
               if (QP_betas(i, 1) < C_value-(1e-5))
                   UBSVs=UBSVs+1;
               end
            end
            QP_betas_sum=QP_betas_sum+QP_betas(i, 1);   
        end

        %storing all the SVs inside an array called supp_vectors
        supp_vectors=zeros(support_vector_count, dimension_patterns+2);
        support_vector_count=1;
        first_UBSV_index=0;
        
        for i=1:1:pattern_typical_count
            if (QP_betas(i, 1) > 1e-5)
                for k=1:1:dimension_patterns
                    supp_vectors(support_vector_count, k)=X(i, k);
                end
                if (QP_betas(i, 1) < C_value-(1e-5) & first_UBSV_index == 0)
                    first_UBSV_index=i;
                end
                supp_vectors(support_vector_count, dimension_patterns+1)=QP_betas(i, 1);
                supp_vectors(support_vector_count, dimension_patterns+2)=i;
                support_vector_count=support_vector_count+1;
            end
        end
      
       
      support_vector_count=support_vector_count-1;
        %to calculate R and maintain as a reference
        
        R=1.0;
        cross_product=zeros(pattern_typical_count, 1);

        for i=1:1:pattern_typical_count
            
            if(QP_betas(i, 1) > 1e-5)
                R=R-QP_betas(i, 1)*Q(first_UBSV_index, i);
            end
            
            for j=1:1:pattern_typical_count
                cross_product(i, 1)=cross_product(i, 1)+QP_betas(i, 1)*QP_betas(j, 1)*(0.5*Q(i, j));
            end
            
            if (QP_betas(i, 1) > 1e-5)
                R=R+cross_product(i, 1);
            end
            
        end
        
      
        
        c=(1-R);    %recall that R here is actually R^2 as per the publication
        for j=1:1:pattern_typical_count
            c=c+cross_product(j, 1);
        end
        
        %testing accuracy w.r.t. second filename
        testingfile=fopen(testing_filename, 'r');
        testing_typical_count=0;
        testing_patterns=fscanf(testingfile, '%d', 1);
        dimension_patterns=fscanf(testingfile, '%d', 1);
        misclassifieds=0;
        
        for i=1:1:testing_patterns
            for k=1:1:dimension_patterns
                pattern_point(k, 1)=fscanf(testingfile, '%f', 1);
            end

            pattern_type=fscanf(testingfile, '%d', 1);

            c_temp=c;
            for j=1:1:pattern_typical_count
                if (QP_betas(j, 1) > 1e-5)
                    norm_2=norm(pattern_point(:, 1)-X(j,:)')^2;
                    c_temp=c_temp-2.0*QP_betas(j, 1)*exp(-gamma_value*norm_2);
                end
            end
            if (pattern_type == 1) testing_typical_count=testing_typical_count+1; end

            if (c_temp>0)
                if (pattern_type == 1)
                    misclassifieds=misclassifieds+1;
                end
            else
                if (pattern_type == -1)
                    misclassifieds=misclassifieds+1;
                end
            end
        end
        
        misclassifieds=misclassifieds/testing_patterns;

        true_misclassifieds=min(true_misclassifieds, misclassifieds);
        
        fprintf(1, 'CSVDD: %0.3f typ/test: %d/%d, typ/train %d/%d SVs: %d UBSVs: %d fopt: %0.2f', 1.0-misclassifieds, testing_typical_count, testing_patterns, pattern_typical_count, total_patterns, support_vector_count, UBSVs, fopt);
        
        misclassifieds_array(II, JJ, 1)=misclassifieds;
                        
        fclose(testingfile);
                 
        %agent's pre-image calculation using formula in (31)
        alpha_K_alpha=(QP_betas')*(0.5*Q)*QP_betas;
        
        gamma_SVDD=1.0/((alpha_K_alpha)^(0.5));
        
        agent_preimage=zeros(1, dimension_patterns);
        for i=1:1:pattern_typical_count
            agent_preimage=agent_preimage+(1.0/alpha_K_alpha)*(QP_betas(i, 1))*(QP_betas'*(0.5*Q(i, :)'))*X(i, :);
        end

        for i=1:1:dimension_patterns
            misclassifieds_array(II, JJ, i+1)=agent_preimage(1, i);
        end
        
        C_prime_SVDD=1-R+1.0/((gamma_SVDD)^2.0);
        
        misclassifieds_array(II, JJ, dimension_patterns+2)=(R^0.5);
        
        %testing accuracy using approximation method
        testingfile=fopen(testing_filename, 'r');
        
        testing_typical_count=0;
        
        testing_patterns=fscanf(testingfile, '%d', 1);
        dimension_patterns=fscanf(testingfile, '%d', 1);
        misclassifieds=0;
        for i=1:1:testing_patterns
            for j=1:1:dimension_patterns
                pattern_point(j, 1)=fscanf(testingfile, '%f', 1);
            end

            pattern_type=fscanf(testingfile, '%d', 1);
            
            if (pattern_type == 1)
                testing_typical_count=testing_typical_count+1;
            end
            
            norm_2=norm(pattern_point(:, 1)-agent_preimage(1, :)')^2;
            
            decision_SVDD=C_prime_SVDD-(2.0/gamma_SVDD)*exp(-gamma_value*norm_2);

            if ( decision_SVDD > 0.0 )
                 if(pattern_type == 1)
                     misclassifieds=misclassifieds+1;
                 end
            else
                if ( pattern_type == -1 )
                    misclassifieds=misclassifieds+1;
                end
            end
        end
        
        misclassifieds=misclassifieds/testing_patterns;

        true_misclassifieds=min(true_misclassifieds, misclassifieds);

        trainingfile=fopen(patterns_filename, 'r');
        training_patterns=fscanf(trainingfile, '%d', 1);
        dimension_patterns=fscanf(trainingfile, '%d', 1);
        misclassifieds_training=0;
        
        for i=1:1:training_patterns
            for k=1:1:dimension_patterns
                pattern_point(k, 1)=fscanf(trainingfile, '%f', 1);
            end

            pattern_type=fscanf(trainingfile, '%d', 1);

            c_temp=c;
            for j=1:1:pattern_typical_count
                norm_2=norm(pattern_point(:, 1)-X(j,:)')^2;
                c_temp=c_temp-2.0*QP_betas(j, 1)*exp(-gamma_value*norm_2);
            end

            if (c_temp>0)
                if (pattern_type == 1)
                    misclassifieds_training=misclassifieds_training+1;
                end
            else
                if (pattern_type == -1)
                    misclassifieds_training=misclassifieds_training+1;
                end
            end
        end
        
        misclassifieds_training=misclassifieds_training/training_patterns;

        true_misclassifieds=min(true_misclassifieds, misclassifieds_training);
        fclose(trainingfile);

        testingfile=fopen(testing_filename, 'r');
        testing_patterns=fscanf(testingfile, '%d', 1);
        dimension_patterns=fscanf(testingfile, '%d', 1);
        misclassifieds_testing=0;
        
        for i=1:1:testing_patterns
            for k=1:1:dimension_patterns
                pattern_point(k, 1)=fscanf(testingfile, '%f', 1);
            end

            pattern_type=fscanf(testingfile, '%d', 1);
            norm_2=norm(pattern_point(:, 1)-agent_preimage(1, :)')^2;
            
            radius_squared=2.0-2.0*exp(-gamma_value*norm_2);

            if ( radius_squared-R > 0)
                if (pattern_type == 1)
                    misclassifieds_testing=misclassifieds_testing+1;
                end
            else
                if (pattern_type == -1)
                    misclassifieds_testing=misclassifieds_testing+1;
                end
            end
        end
        
        misclassifieds_testing=misclassifieds_testing/testing_patterns;

        true_misclassifieds=min(true_misclassifieds, misclassifieds_testing);
        fclose(testingfile);
        
        fprintf(1, ' FSVDD: %0.3f CSVDD-train: %0.3f FSVDD-ala-C: %0.3f\n#define C_FROM_QP %0.2f \n#define SIGMA_SQUARED %0.5f \ndouble scilab_centre[]={', 1.0-misclassifieds, 1.0-misclassifieds_training, 1.0-misclassifieds_testing, C_value, 1.0/(2.0*gamma_value));
        sigma_squared=1.0/(2.0*gamma_value);
        
        for k=1:1:dimension_patterns
            if k < dimension_patterns
                fprintf(1, '%0.3f, ', agent_preimage(1, k));
            else
                fprintf(1, '%0.3f};', agent_preimage(1, k));
            end            
        end

        %attempting the modified SVDD calculation method using a' and R' as
        %starting points to infer a and R
        
        R_prime=(R+(1.0-1.0/gamma_SVDD)^2)^(0.5);
        K_prime=exp(-(1.0/(2.0*sigma_squared))*(-2.0*sigma_squared*log((2.0-(R_prime)^2)/2.0)));
        
        fprintf(1, '\ndouble scilab_radius=%0.3f;', R_prime);

        fprintf(1, '\ndouble SVDD_border_point[]={');
        
        for k=1:1:dimension_patterns
            if k < dimension_patterns
                fprintf(1, '%0.3f, ', agent_preimage(1, k));
            else
                fprintf(1, '%0.3f};', agent_preimage(1, k)+sqrt(K_prime));
            end            
        end
        
        testingfile=fopen(testing_filename, 'r');
        testing_patterns=fscanf(testingfile, '%d', 1);
        dimension_patterns=fscanf(testingfile, '%d', 1);
        misclassifieds_testing=0;
        
        for i=1:1:testing_patterns
            for k=1:1:dimension_patterns
                pattern_point(k, 1)=fscanf(testingfile, '%f', 1);
            end

            pattern_type=fscanf(testingfile, '%d', 1);
            norm_2=norm(pattern_point(:, 1)-agent_preimage(1, :)')^2;
            
            radius_squared=1.0+((K_prime)^2.0)-2.0*K_prime*exp(-gamma_value*norm_2);

            if ( radius_squared-R > 0)
                if (pattern_type == 1)
                    misclassifieds_testing=misclassifieds_testing+1;
                end
            else
                if (pattern_type == -1)
                    misclassifieds_testing=misclassifieds_testing+1;
                end
            end
        end
        
        misclassifieds_testing=misclassifieds_testing/testing_patterns;

        true_misclassifieds=min(true_misclassifieds, misclassifieds_testing);
        fclose(testingfile);
        
        fprintf(1, '\nAccuracy in primal a''-based SVDD=%0.3f,\n', 1.0-misclassifieds_testing);
        else
            
            fprintf(1, '\nerror condition in quadprog: %d (-6 means non-convex problem, -2 means infeasible problem)', lagr);
            
        end
        end
        end
end

