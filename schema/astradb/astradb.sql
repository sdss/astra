create schema astra;

set search_path to astra;

drop table if exists astra.task cascade;
drop table if exists astra.parameter cascade;
drop table if exists astra.task_parameter  cascade;
drop table if exists astra.batch_interface cascade;
drop table if exists astra.output_interface cascade;

/* Contributed methods */
drop table if exists astra.ferre cascade;
drop table if exists astra.doppler cascade;
drop table if exists astra.thecannon cascade;
drop table if exists astra.apogeenet cascade;
drop table if exists astra.classification cascade;
drop table if exists astra.thepayne cascade;
drop table if exists astra.wd_classification cascade;
drop table if exists astra.aspcap cascade;

/* Create the interface table for outputs from different methods */
create table astra.output_interface (
    pk serial primary key
);

/* Create the primary task table */
create table astra.task (
    pk serial primary key,
    task_module text not null,
    task_id text not null,
    status_code int default 0,
    output_pk bigint,
    duration real,
    created timestamp default now(),
    modified timestamp default now(),
    foreign key (output_pk) references astra.output_interface(pk) on delete cascade
);
create unique index on astra.task (task_module, task_id);

/* Create tables for contributed methods */

/* Doppler */
create table astra.doppler (
    output_pk int primary key,
    vhelio real[],
    vrel real[],
    u_vrel real, /* Same as `vrelerr`, changed to be consistent with other astra tables */
    teff real[],
    u_teff real[], /* Same as `tefferr` */
    logg real[],
    u_logg real[],
    fe_h real[],
    u_fe_h real[],
    chisq real[],
    bc real[],
    foreign key (output_pk) references astra.output_interface(pk) on delete restrict
);

/* APOGEENet */
create table astra.apogeenet (
    output_pk int primary key,
    snr real[],
    teff real[],
    u_teff real[],
    logg real[],
    u_logg real[],
    fe_h real[],
    u_fe_h real[],
    bitmask_flag int[],
    foreign key (output_pk) references astra.output_interface(pk) on delete restrict
);

/* FERRE */
create table astra.ferre (
    output_pk int primary key,
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
    foreign key (output_pk) references astra.output_interface(pk) on delete restrict
);

/* ASPCAP (final results from many FERRE executions) */
create table astra.aspcap (
    output_pk int primary key,
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
    foreign key (output_pk) references astra.output_interface(pk) on delete restrict
);

/* The Cannon 

NOTE: If the label names ever change between models of The Cannon, then we will either 
      need versioned tables (like thecannon_v1) or we will need to re-think this schema.
*/
create table astra.thecannon (
    output_pk int primary key,
    teff real,
    foreign key (output_pk) references astra.output_interface(pk) on delete restrict
);

/* 
The Payne 

NOTE: If the label names ever change between models of The Payne, then we will either 
      need versioned tables (like thepayne_v1) or we will need to re-think this schema.
*/
create table astra.thepayne (
    output_pk int primary key,
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
    foreign key (output_pk) references astra.output_interface(pk) on delete restrict
);

/* Classifiers */
create table astra.classification (
    output_pk int primary key,
    log_prob real[],
    prob real[],
    foreign key (output_pk) references astra.output_interface(pk) on delete restrict
);

/* Classifiers specifically for white dwarfs */
create table astra.wd_classification (
    output_pk int primary key,
    wd_class char(2),
    flag boolean,
    foreign key (output_pk) references astra.output_interface(pk) on delete restrict
);


/* Create the parameter table */
create table astra.parameter (
    pk serial primary key,
    parameter_name text,
    parameter_value text
);
create unique index unique_parameter on astra.parameter (parameter_name, parameter_value);

/* Create the junction table between tasks and parameters */
create table astra.task_parameter (
    pk serial primary key,
    task_pk bigint,
    parameter_pk bigint,
    foreign key (parameter_pk) references astra.parameter(pk) on delete restrict,
    foreign key (task_pk) references astra.task(pk) on delete restrict
);
create unique index task_parameter_ref on astra.task_parameter (task_pk, parameter_pk);

/* Create the interface for batched tasks */
create table astra.batch_interface (
    pk serial primary key,
    parent_task_pk bigint not null,
    child_task_pk bigint not null,
    foreign key (child_task_pk) references astra.task(pk) on delete restrict,
    foreign key (parent_task_pk) references astra.task(pk) on delete restrict
);
create unique index on astra.batch_interface (parent_task_pk, child_task_pk);