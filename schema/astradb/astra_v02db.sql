create schema astra_v02;

set search_path to astra_v02;

drop table if exists astra_v02.ti cascade;
drop table if exists astra_v02.parameter cascade;
drop table if exists astra_v02.ti_parameter cascade;
drop table if exists astra_v02.output_interface cascade;

/* Contributed methods */
drop table if exists astra_v02.ferre cascade;
drop table if exists astra_v02.doppler cascade;
drop table if exists astra_v02.thecannon cascade;
drop table if exists astra_v02.apogeenet cascade;
drop table if exists astra_v02.classification cascade;
drop table if exists astra_v02.thepayne cascade;
drop table if exists astra_v02.wd_classification cascade;
drop table if exists astra_v02.aspcap cascade;

/* Create the interface table for outputs from different methods */
create table astra_v02.output_interface (
    pk serial primary key
);


/* Create the primary task instance table */
create table astra_v02.ti (
    pk serial primary key,
    dag_id text,
    task_id text not null,
    run_id text,
    output_pk bigint,
    created timestamp default now(),
    foreign key (output_pk) references astra_v02.output_interface(pk) on delete cascade
);

/* Create tables for contributed methods */

/* Doppler */
create table astra_v02.doppler (
    output_pk int primary key,
    ti_pk bigint,
    vhelio real[],
    vrel real[],
    u_vrel real, /* Same as `vrelerr`, changed to be consistent with other astra_v02 tables */
    teff real[],
    u_teff real[], /* Same as `tefferr` */
    logg real[],
    u_logg real[],
    fe_h real[],
    u_fe_h real[],
    chisq real[],
    bc real[],
    foreign key (output_pk) references astra_v02.output_interface(pk) on delete restrict
);

/* APOGEENet */
create table astra_v02.apogeenet (
    output_pk int primary key,
    ti_pk bigint,
    snr real[],
    teff real[],
    u_teff real[],
    logg real[],
    u_logg real[],
    fe_h real[],
    u_fe_h real[],
    bitmask_flag int[],
    foreign key (output_pk) references astra_v02.output_interface(pk) on delete restrict
);

/* FERRE */
create table astra_v02.ferre (
    output_pk int primary key,
    ti_pk bigint,
    snr real[],
    frozen_teff boolean,
    frozen_logg boolean,
    frozen_metals boolean,
    frozen_log10vdop boolean,
    frozen_o_mg_si_s_ca_ti boolean,
    frozen_lgvsini boolean,
    frozen_c boolean,
    frozen_n boolean,
    initial_teff real[],
    initial_logg real[],
    initial_metals real[],
    initial_log10vdop real[],
    initial_o_mg_si_s_ca_ti real[],
    initial_lgvsini real[],
    initial_c real[],
    initial_n real[],
    teff real[],
    u_teff real[],
    logg real[],
    u_logg real[],
    metals real[],
    u_metals real[],
    log10vdop real[],
    u_log10vdop real[],
    o_mg_si_s_ca_ti real[],
    u_o_mg_si_s_ca_ti real[],
    lgvsini real[],
    u_lgvsini real[],
    c real[],
    u_c real[],
    n real[],
    u_n real[],
    log_chisq_fit real[],
    log_snr_sq real[],
    bitmask_flag int[],
    foreign key (output_pk) references astra_v02.output_interface(pk) on delete restrict
);

/* ASPCAP (final results from many FERRE executions) */
create table astra_v02.aspcap (
    output_pk int primary key,
    ti_pk bigint,
    snr real[],
    teff real[],
    u_teff real[],
    logg real[],
    u_logg real[],
    metals real[],
    u_metals real[],
    log10vdop real [],
    u_log10vdop real[],
    o_mg_si_s_ca_ti real[],
    u_o_mg_si_s_ca_ti real[],
    lgvsini real[],
    u_lgvsini real[],
    c_h real[],
    u_c_h real[],
    n_h real[],
    u_n_h real[],
    cn_h real[],
    u_cn_h real[],
    al_h real[],
    u_al_h real[],
    ca_h real[],
    u_ca_h real[],
    ce_h real[],
    u_ce_h real[],
    co_h real[],
    u_co_h real[],
    cr_h real[],
    u_cr_h real[],
    cu_h real[],
    u_cu_h real[],
    fe_h real[],
    u_fe_h real[],
    mg_h real[],
    u_mg_h real[],
    mn_h real[],
    u_mn_h real[],
    na_h real[],
    u_na_h real[],
    nd_h real[],
    u_nd_h real[],
    ni_h real[],
    u_ni_h real[],
    o_h real[],
    u_o_h real[],
    p_h real[],
    u_p_h real[],
    rb_h real[],
    u_rb_h real[],
    si_h real[],
    u_si_h real[],
    s_h real[],
    u_s_h real[],
    ti_h real[],
    u_ti_h real[],
    v_h real[],
    u_v_h real[],
    yb_h real[],
    u_yb_h real[],    
    foreign key (output_pk) references astra_v02.output_interface(pk) on delete restrict
);

/* The Cannon 

NOTE: If the label names ever change between models of The Cannon, then we will either 
      need versioned tables (like thecannon_v1) or we will need to re-think this schema.
*/
create table astra_v02.thecannon (
    output_pk int primary key,
    ti_pk bigint,
    teff real,
    foreign key (output_pk) references astra_v02.output_interface(pk) on delete restrict
);

/* 
The Payne 

NOTE: If the label names ever change between models of The Payne, then we will either 
      need versioned tables (like thepayne_v1) or we will need to re-think this schema.
*/
create table astra_v02.thepayne (
    output_pk int primary key,
    ti_pk bigint,
    snr real[],
    teff real[],
    u_teff real[],
    logg real[],
    u_logg real[],
    v_turb real[],
    u_v_turb real[],
    c_h real[],
    u_c_h real[],
    n_h real[],
    u_n_h real[],
    o_h real[],
    u_o_h real[],
    na_h real[],
    u_na_h real[],
    mg_h real[],
    u_mg_h real[],
    al_h real[],
    u_al_h real[],
    si_h real[],
    u_si_h real[],
    p_h real[],
    u_p_h real[],
    s_h real[],
    u_s_h real[],
    k_h real[],
    u_k_h real[],
    ca_h real[],
    u_ca_h real[],
    ti_h real[],
    u_ti_h real[],
    v_h real[],
    u_v_h real[],
    cr_h real[],
    u_cr_h real[],
    mn_h real[],
    u_mn_h real[],
    fe_h real[],
    u_fe_h real[],
    co_h real[],
    u_co_h real[],
    ni_h real[],
    u_ni_h real[],
    cu_h real[],
    u_cu_h real[],
    ge_h real[],
    u_ge_h real[],
    c12_c13 real[],
    u_c12_c13 real[],
    v_macro real[],
    u_v_macro real[],
    bitmask_flag int[],
    foreign key (output_pk) references astra_v02.output_interface(pk) on delete restrict
);

/* Classifiers */
create table astra_v02.classification (
    output_pk int primary key,
    ti_pk bigint,
    log_prob real[],
    prob real[],
    foreign key (output_pk) references astra_v02.output_interface(pk) on delete restrict
);

/* Classifiers specifically for white dwarfs */
create table astra_v02.wd_classification (
    output_pk int primary key,
    ti_pk bigint,
    wd_class char(2),
    flag boolean,
    foreign key (output_pk) references astra_v02.output_interface(pk) on delete restrict
);

/* Create the parameter table */
create table astra_v02.parameter (
    pk serial primary key,
    parameter_name text,
    parameter_value text
);
create unique index unique_parameter on astra_v02.parameter (parameter_name, parameter_value);

/* Create the junction table between tasks and parameters */
create table astra_v02.ti_parameter (
    pk serial primary key,
    ti_pk bigint not null,
    parameter_pk bigint not null,
    foreign key (parameter_pk) references astra_v02.parameter(pk) on delete restrict,
    foreign key (ti_pk) references astra_v02.ti(pk) on delete restrict
);
create unique index ti_parameter_ref on astra_v02.ti_parameter (ti_pk, parameter_pk);

