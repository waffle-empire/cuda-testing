#pragma once
#include <cinttypes>

struct measurement
{
    int m_ttf;
    int m_engine_id;
    int m_cycle;
    int m_bleed_enthalpy;
    
    float m_setting1;
    float m_setting2;
    float m_temp_lpc_outlet;
    float m_temp_hpc_outlet;
    float m_temp_lpt_outlet;
    float m_pressure_hpc_outlet;
    float m_physical_fan_speed;
    float m_physical_core_speed;
    float m_static_pressure_hpc_outlet;
    float m_fuel_flow_ration_ps30;
    float m_corrected_fan_speed;
    float m_corrected_core_speed;
    float m_bypass_ratio;
    float m_hpt_collant_bleed;
    float m_lpt_coolant_bleed;

    measurement() = default;
    measurement(int engine_id, int cycle, float setting1, float setting2, float temp_lpc_outlet, float temp_hpc_outlet, float temp_lpt_outlet, float pressure_hpc_outlet, float physical_fan_speed, float physical_core_speed, float static_pressure_hpc_outlet, float fuel_flow_ration_ps30, float corrected_fan_speed, float corrected_core_speed, float bypass_ratio, int bleed_enthalpy, float hpt_collant_bleed, float lpt_coolant_bleed, int ttf)
    {
        m_ttf = ttf;
        m_engine_id = engine_id;
        m_cycle = cycle;
        m_bleed_enthalpy = bleed_enthalpy;
        
        m_setting1 = setting1;
        m_setting2 = setting2;
        m_temp_lpc_outlet = temp_lpc_outlet;
        m_temp_hpc_outlet = temp_hpc_outlet;
        m_temp_lpt_outlet = temp_lpt_outlet;
        m_pressure_hpc_outlet = pressure_hpc_outlet;
        m_physical_fan_speed = physical_fan_speed;
        m_physical_core_speed = physical_core_speed;
        m_static_pressure_hpc_outlet = static_pressure_hpc_outlet;
        m_fuel_flow_ration_ps30 = fuel_flow_ration_ps30;
        m_corrected_fan_speed = corrected_fan_speed;
        m_corrected_core_speed = corrected_core_speed;
        m_bypass_ratio = bypass_ratio;
        m_hpt_collant_bleed = hpt_collant_bleed;
        m_lpt_coolant_bleed = lpt_coolant_bleed;
    }

};