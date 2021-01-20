create schema astra;

set search_path to astra;

drop table if exists astra.task_state cascade;
drop table if exists astra.task_parameter cascade;
drop table if exists astra.apogee_visit cascade;
drop table if exists astra.apogee_star cascade;
drop table if exists astra.boss_spec cascade;
drop table if exists astra.classification cascade;
drop table if exists astra.classification_class cascade;
drop table if exists astra.apogeenet cascade;
drop table if exists astra.thepayne cascade;
drop table if exists astra.ferre cascade;
drop table if exists astra.aspcap cascade;

create table astra.task_state (
    pk serial primary key not null,
    task_module text not null,
    task_id text not null,
    parameter_pk bigint,
    status_code int default -1,
    created timestamp default now(),
    completed timestamp,
    modified timestamp default now(),
    duration real
);
create unique index task_ref on astra.task_state (task_module, task_id);


create table astra.task_parameter (
    pk bigint primary key not null,
    parameters jsonb
);


create table astra.apogee_visit (
    pk serial primary key not null,
    catalog_id bigint,
    release text not null default 'sdss5',
    apred text not null,
    mjd int not null,
    telescope text not null,
    field text not null,
    plate text not null,
    fiber int not null
);

create table astra.apogee_star (
    pk serial primary key not null,
    catalog_id bigint,
    apogee_visit_pk bigint[],
    release text not null default 'sdss5',
    apred text not null,
    telescope text not null,
    healpix int not null,
    obj text not null
);

create table astra.boss_spec (
    pk serial primary key not null
);

create table astra.classification (
    pk serial primary key not null,
    task_pk bigint,
    class_pk int[],
    log_prob real[]
);

create table astra.classification_class (
    pk serial primary key not null,
    description text
);


create table astra.thepayne (
    pk serial primary key not null,
    task_pk bigint,
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
    bitmask_flag int[]
);


create table astra.ferre (
    pk serial primary key not null,
    task_pk bigint,
    snr real[],
    initial_teff real[],
    initial_logg real[],
    initial_metals real[],
    initial_log10vdop real[],
    initial_o_mg_si_s_ca_ti real[],
    initial_lgvsini real[],
    initial_c real[],
    initial_n real[],
    frozen_teff real[],
    frozen_logg real[],
    frozen_metals real[],
    frozen_log10vdop real[],
    frozen_o_mg_si_s_ca_ti real[],
    frozen_lgvsini real[],
    frozen_c real[],
    frozen_n real[],
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
    chisq real[],
    snr_fit real[],
    bitmask_flag int[]
);

create table astra.apogeenet (
    pk serial primary key not null,
    task_pk bigint,
    snr real[],
    teff real[],
    u_teff real[],
    logg real[],
    u_logg real[],
    fe_h real[],
    u_fe_h real[],
    bitmask_flag int[]
);



create table astra.aspcap (
    pk serial primary key not null,
    task_pk bigint,
    ferre_pk bigint[],
    snr real[],
    teff real[],
    u_teff real[],
    logg real[],
    u_logg real[],
    fe_h real[],
    u_fe_h real[],
    log10vdop real[],
    u_log10vdop real[],
    lgvsini real[],
    u_lgvsini real[],
    al_h real[],
    u_al_h real[],
    ca_h real[],
    u_ca_h real[],
    ce_h real[],
    u_ce_h real[],
    c_h real[],
    u_c_h real[],
    cn_h real[],
    u_cn_h real[],
    co_h real[],
    u_co_h real[],
    cr_h real[],
    u_cr_h real[],
    cu_h real[],
    u_cu_h real[],
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
    n_h real[],
    u_n_h real[],
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
    chisq real[],
    bitmask_flag int[]
);



alter table only astra.aspcap
    add constraint task_fk
    foreign key (task_pk) references astra.task_state(pk)
    on delete cascade;

alter table only astra.apogeenet
    add constraint task_fk
    foreign key (task_pk) references astra.task_state(pk)
    on delete cascade;

alter table only astra.ferre
    add constraint task_fk
    foreign key (task_pk) references astra.task_state(pk)
    on delete cascade;

alter table only astra.thepayne
    add constraint task_fk
    foreign key (task_pk) references astra.task_state(pk)
    on delete cascade;

alter table only astra.classification
    add constraint task_fk
    foreign key (task_pk) references astra.task_state(pk)
    on delete cascade;
    